"""This script aims to repath original tiles and metadata to point to the normalized slides."""

import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


def repath_and_log(
    tiling_uri: str, normalized_map: dict[str, str], dataset_name: str
) -> None:
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    slides["path"] = slides["path"].map(lambda x: normalized_map[Path(x).stem])
    save_mlflow_dataset(slides, tiles, f"{dataset_name}_normalized")


def repath_metadata_and_log(
    metadata_uri: str, normalized_map: dict[str, str], logger: MLFlowLogger
) -> None:
    metadata_path = Path(mlflow.artifacts.download_artifacts(metadata_uri))
    metadata = pd.read_csv(metadata_path)

    metadata["slide_path"] = metadata["slide_path"].map(
        lambda x: normalized_map[Path(x).stem]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / f"{metadata_path.stem}_normalized.csv"
        metadata.to_csv(out, index=False)
        logger.log_artifact(str(out))


@with_cli_args(["+preprocessing=repath_normalization"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    normalized_slides = list(Path(config.normalized_data_dir).glob("*.tiff"))
    normalized_map = {p.stem: str(p) for p in normalized_slides}

    if config.data.get("metadata_table") is not None:
        repath_metadata_and_log(config.data.metadata_table, normalized_map, logger)

    if (
        hasattr(config.data, "tiles_filtered_uri_512")
        and config.data.tiles_filtered_uri_512 is not None
    ):
        repath_and_log(
            config.data.tiles_filtered_uri_512,
            normalized_map,
            config.data.data_name + "512",
        )

    if (
        hasattr(config.data, "tiles_filtered_uri_224")
        and config.data.tiles_filtered_uri_224 is not None
    ):
        repath_and_log(
            config.data.tiles_filtered_uri_224,
            normalized_map,
            config.data.data_name + "224",
        )


if __name__ == "__main__":
    main()
