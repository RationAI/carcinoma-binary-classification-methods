"""Script to convert old way of handling embeddings (separate embeddings file per slide) to the new one where they are included in parquet files."""

import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray.data import Dataset, SaveMode


def resolve_embeddings_dir(config: DictConfig) -> Path:
    uri = config.filtered_embeddings_uri
    path = config.filtered_embeddings_path

    if uri is None and path is None:
        raise ValueError(
            "Either `filtered_embeddings_uri` or `filtered_embeddings_path` "
            "must be provided."
        )

    if path is not None and (uri is None or config.use_filtered_embeddings_path):
        return Path(path)

    return Path(mlflow.artifacts.download_artifacts(uri))


def tag_row_order(src: Path, dst: Path, batch_size: int = 1_000_000) -> None:
    """Streams `src` into `dst`, adding a `_row_order` column with each row's original position in the file.

    Embeddings are stored per-slide in the same tile order as the source
    tiles.parquet file, but the groupby/shuffle below doesn't guarantee it
    preserves row order, so `_row_order` lets `attach_embeddings` restore it
    before pairing rows up with their embedding tensor. Done as a streaming
    pass (bounded by `batch_size`, not total row count) so it works the same
    way regardless of how large the tiles table is.
    """
    reader = pq.ParquetFile(src)
    writer = None
    row_order = 0
    try:
        for batch in reader.iter_batches(batch_size=batch_size):
            n = batch.num_rows
            batch = batch.append_column(
                "_row_order", pa.array(range(row_order, row_order + n), type=pa.int64())
            )
            if writer is None:
                writer = pq.ParquetWriter(dst, batch.schema)
            writer.write_batch(batch)
            row_order += n
    finally:
        if writer is not None:
            writer.close()


def attach_embeddings(slide_tiles: pd.DataFrame, embeddings_dir: Path) -> pd.DataFrame:
    assert slide_tiles["path"].nunique() == 1, "Expected one unique path per slide"

    slide_path = slide_tiles["path"].iloc[0]
    slide_name = Path(slide_path).stem

    embeds = (
        torch.load(
            str(embeddings_dir / f"{slide_name}.pt"),
            map_location="cpu",
        )
        .cpu()
        .numpy()
    )

    if len(slide_tiles) != len(embeds):
        raise ValueError(
            f"Mismatch: {len(slide_tiles)} tiles vs {len(embeds)} embeddings for {slide_name}"
        )

    slide_tiles = slide_tiles.sort_values("_row_order").reset_index(drop=True)
    slide_tiles["embedding"] = embeds.tolist()
    return slide_tiles.drop(columns=["path", "_row_order"])


def process_and_shard_tiles(
    tiles_path: Path,
    slides: pd.DataFrame,
    output_dir: Path,
    embeddings_dir: Path,
    rows_per_file: int,
    override_num_blocks: int,
    block_memory_bytes: int,
    concurrency: int | None = None,
) -> None:
    tiles_output = output_dir / "tiles"
    tiles_output.mkdir(parents=True, exist_ok=True)

    slide_info = slides.set_index("id")[["path"]]

    def enrich(batch: pd.DataFrame) -> pd.DataFrame:
        return batch.join(slide_info, on="slide_id")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tagged_tiles_path = Path(tmp_dir) / "tiles_with_row_order.parquet"
        tag_row_order(tiles_path, tagged_tiles_path)

        # read directly off disk (like tile_embeddings_v2.py) instead of loading the
        # whole table into the driver's pandas memory first -- `override_num_blocks`
        # keeps individual blocks small, and the `memory` remote arg tells Ray how
        # much each read task needs so it schedules only as many concurrently as
        # actually fit, giving steady progress instead of OOM-ing
        ds: Dataset = ray.data.read_parquet(
            str(tagged_tiles_path),
            override_num_blocks=override_num_blocks,
            ray_remote_args={"memory": block_memory_bytes},
        )
        ds = ds.map_batches(enrich, batch_format="pandas")

        # gather each slide's tiles together to align with its precomputed embedding
        # tensor; capping the task pool size limits how many slides' embedding
        # tensors get loaded into memory at once, trading off max parallelism for
        # steadier progress
        ds = ds.groupby("slide_id").map_groups(
            attach_embeddings,  # type: ignore[arg-type]
            fn_kwargs={"embeddings_dir": embeddings_dir},
            batch_format="pandas",
            memory=block_memory_bytes,
            compute=ray.data.TaskPoolStrategy(size=concurrency),
        )

        ds.write_parquet(
            str(tiles_output), max_rows_per_file=rows_per_file, mode=SaveMode.OVERWRITE
        )


@with_cli_args(["+preprocessing=merge_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_path = Path(
        mlflow.artifacts.download_artifacts(config.data.tiles_filtered_uri_224)
    )
    slides = pd.read_parquet(tiling_path / "slides.parquet")

    embeds_dir = resolve_embeddings_dir(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_output = output_dir / "slides"
    slides_output.mkdir(parents=True, exist_ok=True)
    slides_path = slides_output / "slides.parquet"
    slides.to_parquet(slides_path, index=False)  # slides.parquet is not changed

    with ray.init(num_cpus=config.num_cpus):
        process_and_shard_tiles(
            tiling_path / "tiles.parquet",
            slides,
            output_dir,
            embeds_dir,
            config.rows_per_file,
            override_num_blocks=config.override_num_blocks,
            block_memory_bytes=int(config.block_memory_gb * 1024**3),
            concurrency=config.concurrency,
        )

    mlflow.log_artifacts(str(output_dir), config.data.data_name + "_sharded")


if __name__ == "__main__":
    main()
