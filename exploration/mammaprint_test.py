""" Mammaprint Test Data Adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def repair_and_log(tiling_path: Path, suffix: str) -> None:
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")
    slides["carcinoma"] = True # all slides are positive
    save_mlflow_dataset(slides, tiles, f"mammaprint_test_{suffix}")


@hydra.main(
    config_path="../configs",
    config_name="exploration/breast/mammaprint_test",
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

    df["carcinoma"] = True # all slides are positive

    repair_and_log(
        original_tiling_512,
        "512",
    )
    repair_and_log(
        original_tiling_224,
        "224"
    )

    with TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / f"mammaprint_test_metadata.csv"
        df.to_csv(str(target), index=False)
        mlflow.log_artifact(str(target))


if __name__ == "__main__":
    main()
