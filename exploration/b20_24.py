"""B20-24 Data Adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def repair_split_and_log(
    tiling_path: Path, all_slides: list[str], val_slide_paths: list[str], suffix: str
) -> None:
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    # Binarize carcinoma the same way as the CSV metadata (T -> True, N -> False)
    slides["carcinoma"] = slides["carcinoma"].map(lambda x: x == "T")

    # Keep only slides (and their tiles) that are actually referenced in the metadata CSV
    all_slide_mask = slides["path"].isin(all_slides)
    slides = slides[all_slide_mask]
    tiles = tiles[tiles["slide_id"].isin(slides["id"])]

    # slides.path matches slide_path from the CSVs
    val_slide_mask = slides["path"].isin(val_slide_paths)
    val_slide_ids = slides.loc[val_slide_mask, "id"]

    train_slides, val_slides = slides[~val_slide_mask], slides[val_slide_mask]

    val_tile_mask = tiles["slide_id"].isin(val_slide_ids)
    train_tiles, val_tiles = tiles[~val_tile_mask], tiles[val_tile_mask]

    save_mlflow_dataset(slides, tiles, f"b20_24_{suffix}")
    save_mlflow_dataset(train_slides, train_tiles, f"b20_24_train_{suffix}")
    save_mlflow_dataset(val_slides, val_tiles, f"b20_24_val_{suffix}")


@hydra.main(
    config_path="../configs",
    config_name="exploration/breast/b20_24",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.original_metadata))
    original_tiling_512 = Path(
        mlflow.artifacts.download_artifacts(config.original_tiling_512)
    )
    original_tiling_224 = Path(
        mlflow.artifacts.download_artifacts(config.original_tiling_224)
    )

    df["carcinoma"] = df["carcinoma"].map(lambda x: x == "T")  # binarize carcinoma
    df["slide_path"] = df["path"]  # rename path column (unified with prostate)
    df = df.drop(
        [
            "id",
            "path",
            "extent_x",
            "extent_y",
            "tile_extent_x",
            "tile_extent_y",
            "stride_x",
            "stride_y",
            "mpp_x",
            "mpp_y",
            "level",
            "slide_id",
            "total",
            "pos",
            "neg",
        ],
        axis=1,
    )  # remove unnecessary attributes

    val_df = df[
        df["split"] == 1
    ]  # use one fold as validation set (unified with prostate)
    train_df = df[df["split"] != 1]  # use rest as training set

    repair_split_and_log(
        original_tiling_512,
        df["slide_path"].tolist(),
        val_df["slide_path"].tolist(),
        "512",
    )
    repair_split_and_log(
        original_tiling_224,
        df["slide_path"].tolist(),
        val_df["slide_path"].tolist(),
        "224",
    )

    with TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "b20_24_metadata.csv"
        df.to_csv(str(target), index=False)

        train_target = Path(tmp_dir) / "b20_24_train_metadata.csv"
        train_df.to_csv(str(train_target), index=False)

        val_target = Path(tmp_dir) / "b20_24_val_metadata.csv"
        val_df.to_csv(str(val_target), index=False)

        mlflow.log_artifact(str(target))
        mlflow.log_artifact(str(train_target))
        mlflow.log_artifact(str(val_target))


if __name__ == "__main__":
    main()
