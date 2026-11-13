"""DCIS Coverage Injection.

The BRACS tiles were created without DCIS coverage. A later tiling of the same
slides adds `dcis_roi_percentage`.

Here, the existing tiles are kept as they are and only `dcis_roi_percentage` is
copied over from the DCIS tiling, matched by (slide, x, y). Embeddings and other
attributes are kept from the original.
"""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


DCIS_COL = "dcis_roi_percentage"


def with_slide_path(tiles: pd.DataFrame, slides: pd.DataFrame) -> pd.DataFrame:
    """Adds the slide path, so tiles of two tilings can be matched by (path, x, y)."""
    return tiles.merge(
        slides[["id", "path"]], left_on="slide_id", right_on="id", validate="m:1"
    ).drop(columns="id")


@with_cli_args(["+correction=inject_dcis"])
@hydra.main(config_path="../configs", config_name="correction", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata = pd.read_csv(mlflow.artifacts.download_artifacts(config.source_of_truth))
    normal_paths = set(metadata.loc[metadata["WSI label"] == "N", "slide_path"])

    tiles_dir = Path(mlflow.artifacts.download_artifacts(config.tiles_to_correct))
    slides = pd.read_parquet(tiles_dir / "slides.parquet")
    tiles = pd.read_parquet(tiles_dir / "tiles.parquet")

    dcis_dir = Path(mlflow.artifacts.download_artifacts(config.dcis_source))
    dcis_tiles = with_slide_path(
        pd.read_parquet(dcis_dir / "tiles.parquet"),
        pd.read_parquet(dcis_dir / "slides.parquet"),
    )

    in_normal = dcis_tiles["path"].isin(normal_paths)
    if (dcis_tiles.loc[in_normal, DCIS_COL] > 0).any():
        raise ValueError("DCIS source has DCIS coverage in a normal slide")

    # left join keeps the existing tiles (rows and order) unchanged
    merged = with_slide_path(tiles, slides).merge(
        dcis_tiles[["path", "x", "y", DCIS_COL]],
        on=["path", "x", "y"],
        how="left",
        validate="1:1",
    )
    unmatched = merged[DCIS_COL].isna()
    if not merged.loc[unmatched, "path"].isin(normal_paths).all():
        raise ValueError("Tiles without DCIS coverage found in a non-normal slide")
    merged[DCIS_COL] = merged[DCIS_COL].fillna(0.0)

    for path, n in merged.loc[unmatched, "path"].value_counts().items():
        print(f"{Path(path).stem}: {n} tiles without a match, set to 0.0")
    logger.log_metrics(
        {
            "tiles": len(merged),
            "tiles_unmatched": int(unmatched.sum()),
            "tiles_with_dcis": int((merged[DCIS_COL] > 0).sum()),
        }
    )

    save_mlflow_dataset(
        slides=slides,
        tiles=merged[[*tiles.columns, DCIS_COL]],
        dataset_name=config.dataset_name,
    )


if __name__ == "__main__":
    main()
