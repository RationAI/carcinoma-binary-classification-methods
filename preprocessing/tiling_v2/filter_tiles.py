"""Script for tiles filtering based on configurable tile filters (see tile_filters.py)."""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset

from preprocessing.tiling_v2.tile_filters import TileFilter, apply_filters


def drop_empty_slides(slides: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    kept = slides["id"].isin(tiles["slide_id"].unique())
    print(f"Dropping {(~kept).sum()}/{len(slides)} slides without tiles")
    return slides[kept]


def filter_and_log(
    tiling_uri: str,
    filters: dict[str, TileFilter],
    metadata: pd.DataFrame | None,
    dataset_name: str,
) -> None:
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")
    tiles = apply_filters(tiles, filters, slides, metadata)
    slides = drop_empty_slides(slides, tiles)
    save_mlflow_dataset(slides, tiles, dataset_name)


@with_cli_args(["+preprocessing=filter_tiles"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    # null entries allow disabling a default filter from an experiment config
    filters: dict[str, TileFilter] = {
        name: tile_filter
        for name, tile_filter in hydra.utils.instantiate(config.tile_filters).items()
        if tile_filter is not None
    }
    metadata = (
        pd.read_csv(mlflow.artifacts.download_artifacts(config.data.metadata_table))
        if config.data.get("metadata_table") is not None
        else None
    )

    if hasattr(config.data, "tiles_uri_512") and config.data.tiles_uri_512 is not None:
        filter_and_log(
            config.data.tiles_uri_512,
            filters,
            metadata,
            config.data.data_name + "_512",
        )

    if hasattr(config.data, "tiles_uri_224") and config.data.tiles_uri_224 is not None:
        filter_and_log(
            config.data.tiles_uri_224,
            filters,
            metadata,
            config.data.data_name + "_224",
        )


if __name__ == "__main__":
    main()
