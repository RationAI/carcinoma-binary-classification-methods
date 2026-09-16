"""Script to convert old way of handling embeddings (separate embeddings file per slide) to the new one where they are included in parquet files."""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
import ray
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray.data import DataContext, Dataset, SaveMode


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


def attach_embeddings_group(group: pd.DataFrame, embeddings_dir: Path) -> pd.DataFrame:
    assert group["path"].nunique() == 1, "Expected one unique path per group"

    slide_path = group["path"].iloc[0]
    slide_name = Path(slide_path).stem

    embeds = (
        torch.load(
            str(embeddings_dir / f"{slide_name}.pt"),
            map_location="cpu",
        )
        .cpu()
        .numpy()
    )

    if len(group) != len(embeds):
        raise ValueError(
            f"Mismatch: {len(group)} tiles vs {len(embeds)} embeddings for {slide_name}"
        )

    group = group.copy().sort_values("_row_order").reset_index(drop=True)
    group["embedding"] = embeds.tolist()
    return group


def process_and_shard_tiles(
    slides: pd.DataFrame,
    tiles: pd.DataFrame,
    output_dir: Path,
    embeddings_dir: Path,
    rows_per_file: int,
    max_hash_shuffle_aggregators: int | None = None,
    override_num_blocks: int | None = None,
) -> None:
    tiles_output = output_dir / "tiles"
    tiles_output.mkdir(parents=True, exist_ok=True)

    tiles_enriched = tiles.join(
        slides.set_index("id")[["path"]],
        on="slide_id",
    )

    # embeddings are matched by rows order, which may be violated in parallel group processing
    tiles_enriched["_row_order"] = range(len(tiles_enriched))

    if max_hash_shuffle_aggregators is not None:
        # groupby() below runs a hash-shuffle; by default Ray Data provisions
        # up to min(2 * cluster_cpus, 128) aggregator actors, which can starve
        # the concurrently running map_groups() tasks for CPU slots and stall
        # the whole pipeline ("N out of M aggregators are ready" warning).
        DataContext.get_current().max_hash_shuffle_aggregators = (
            max_hash_shuffle_aggregators
        )

    # from_pandas() puts the whole DataFrame into a single Ray block unless
    # told otherwise, so the downstream shuffle has to partition that one
    # giant block in one task -- easily requiring more memory than the
    # cluster has and stalling forever. Splitting it up front keeps each
    # task's footprint small enough to actually get scheduled.
    ds: Dataset = ray.data.from_pandas(
        tiles_enriched, override_num_blocks=override_num_blocks
    )

    # batch on the level of slides to avoid opening a single embedding file multiple times
    ds = ds.groupby("slide_id").map_groups(
        attach_embeddings_group,  # type: ignore[arg-type]
        fn_kwargs={"embeddings_dir": embeddings_dir},
        batch_format="pandas",
    )

    ds = ds.drop_columns(["path", "_row_order"])

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
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    embeds_dir = resolve_embeddings_dir(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_output = output_dir / "slides"
    slides_output.mkdir(parents=True, exist_ok=True)
    slides_path = slides_output / "slides.parquet"
    slides.to_parquet(slides_path, index=False)  # slides.parquet is not changed

    with ray.init(num_cpus=10):
        process_and_shard_tiles(
            slides,
            tiles,
            output_dir,
            embeds_dir,
            config.rows_per_file,
            max_hash_shuffle_aggregators=config.max_hash_shuffle_aggregators,
            override_num_blocks=config.override_num_blocks,
        )

    mlflow.log_artifacts(str(output_dir), config.data.data_name + "_sharded")


if __name__ == "__main__":
    main()
