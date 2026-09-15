"""Splits a tile-embeddings dataset (`slides.parquet` + a `tiles/` parquet dataset) into per-split datasets by slide membership.

Unlike `preprocessing.negative_split.split_tiling`, which loads the whole
`tiles.parquet` into memory with pandas, this streams the tiles table via PyArrow
so the (potentially huge) embedding vectors are never fully materialized in memory.
For every split it writes a sharded (multi-file, row-bounded) version of the
tiles output.
"""

import shutil
import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


# data config field -> suffix used when naming the split's output artifact
EMBEDDINGS_FIELDS = {
    "tiles_filtered_w_virchow2_uri_224": "virchow2",
    "tiles_filtered_w_pgp_uri_224": "pgp",
}

UNSET_VALUES = (None, "...")


def split_embeddings(
    embeddings_uri: str,
    split_slide_paths: dict[str, set[str]],
    output_root: Path,
    sharded_rows_per_file: int,
) -> dict[str, Path]:
    """Split a huge slides+tiles embeddings dataset by slide path membership.

    For every split this writes a self-contained `slides/` + `tiles/` dataset
    directory with the matching rows, with `tiles/` sharded into multiple
    row-bounded parquet files -- mirroring how other datasets in this repo
    ship a `..._sharded` artifact. The tiles table is streamed via PyArrow
    with filter pushdown, so it is never fully loaded into memory.

    Arguments:
        embeddings_uri (str): MLflow URI of the source embeddings dataset directory
            (must contain a `slides/slides.parquet` file and a `tiles/` parquet
            dataset).
        split_slide_paths (dict[str, set[str]]): Mapping of split name to the set of
            slide `path` values (matching the `slides.parquet` `path` column) that
            belong to that split.
        output_root (Path): Local directory to write the per-split outputs under.
        sharded_rows_per_file (int): Row count per file for the sharded tiles output.

    Returns:
        dict[str, Path]: Mapping of split name to its local output directory. Splits
            with no matching slides are omitted.
    """
    source_path = Path(mlflow.artifacts.download_artifacts(embeddings_uri))
    slides = pd.read_parquet(source_path / "slides" / "slides.parquet")
    tiles_dataset = ds.dataset(source_path / "tiles", format="parquet")

    outputs: dict[str, Path] = {}
    for split_name, slide_paths in split_slide_paths.items():
        split_slides = slides[slides["path"].isin(slide_paths)].reset_index(drop=True)
        if split_slides.empty:
            continue

        split_dir = output_root / split_name
        slides_dir = split_dir / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)
        split_slides.to_parquet(slides_dir / "slides.parquet", index=False)

        row_filter = ds.field("slide_id").isin(pa.array(split_slides["id"].tolist()))

        ds.write_dataset(
            tiles_dataset.scanner(filter=row_filter),
            base_dir=split_dir / "tiles",
            format="parquet",
            basename_template="tiles-{i}.parquet",
            max_rows_per_file=sharded_rows_per_file,
            max_rows_per_group=sharded_rows_per_file,
        )

        outputs[split_name] = split_dir

    return outputs


@with_cli_args(["+preprocessing=split_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    split_slide_paths = {}
    split_data_names = {}
    for split_name, split_data in config.splits.items():
        metadata = pd.read_csv(
            mlflow.artifacts.download_artifacts(split_data.metadata_table)
        )
        split_slide_paths[split_name] = set(metadata["slide_path"])
        split_data_names[split_name] = split_data.data_name

    for field, suffix in EMBEDDINGS_FIELDS.items():
        uri = config.data.get(field)
        if uri in UNSET_VALUES:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = split_embeddings(
                embeddings_uri=uri,
                split_slide_paths=split_slide_paths,
                output_root=Path(tmpdir),
                sharded_rows_per_file=config.sharded_rows_per_file,
            )
            for split_name, split_dir in outputs.items():
                artifact_name = f"{split_data_names[split_name]}_{suffix}_sharded"
                logger.log_artifacts(str(split_dir), artifact_name)
                shutil.rmtree(split_dir)


if __name__ == "__main__":
    main()
