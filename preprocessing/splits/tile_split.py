"""Splits a tiled dataset by slides while keeping the tile-level label proportions.

All tiles of a slide end up in the same split (no leakage), and the proportion of
positive tiles (`label_column` > `label_threshold`) in both splits is kept as close
as possible to the whole dataset.
"""

import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset

from preprocessing.splits.group_split import stratified_group_split


def describe(name: str, tiles: pd.DataFrame, labels: pd.Series) -> None:
    print(
        f"{name}: {tiles['slide_id'].nunique()} slides, {len(tiles)} tiles, "
        f"positive tiles {labels.loc[tiles.index].mean():.3f}"
    )


@with_cli_args(["+preprocessing=tile_split"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert len(config.split_names) == 2, "split_names must be [selected, rest]"
    selected_name, rest_name = config.split_names

    tiling_path = Path(mlflow.artifacts.download_artifacts(config.tiles_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")
    metadata = pd.read_csv(
        mlflow.artifacts.download_artifacts(config.data.metadata_table)
    )

    labels = tiles[config.label_column] > config.label_threshold
    rest_tiles, selected_tiles = stratified_group_split(
        original_data=tiles,
        labels=labels,
        groups=tiles["slide_id"],
        test_size=config.selected_size,
        random_state=42,
    )
    assert not set(selected_tiles["slide_id"]) & set(rest_tiles["slide_id"]), (
        "a slide is present in both splits"
    )

    describe("All", tiles, labels)
    for name, split_tiles in ((selected_name, selected_tiles), (rest_name, rest_tiles)):
        describe(name, split_tiles, labels)

        split_slides = slides[slides["id"].isin(split_tiles["slide_id"])]
        save_mlflow_dataset(
            split_slides, split_tiles, f"{config.data.data_name}_{name}_{config.suffix}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / f"{config.data.data_name}_{name}.csv"
            metadata[metadata["slide_path"].isin(split_slides["path"])].to_csv(
                out, index=False
            )
            logger.log_artifact(str(out))


if __name__ == "__main__":
    main()
