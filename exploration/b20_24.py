""" B20-24 Metadata Adapter. """

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(
    config_path="../configs",
    config_name="exploration/breast/b20_24",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.original_metadata))
    df["carcinoma"] = df["carcinoma"].map(lambda x: x == "T") # binarize carcinoma
    df["slide_path"] = df["path"] # rename path column (unified with prostate)
    df = df.drop(
        ["id", "path", "extent_x", "extent_y", "tile_extent_x", "tile_extent_y", "stride_x", "stride_y", "mpp_x", "mpp_y", "level", "slide_id", "total", "pos", "neg"],
        axis=1
    ) # remove unnecessary attributes

    val_df = df[ df["split"] == 1 ] # use one fold as validation set (unified with prostate)
    train_df = df[ df["split"] != 1 ] # use rest as training set

    with TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / f"b20_24_metadata.csv"
        df.to_csv(str(target), index=False)

        train_target = Path(tmp_dir) / f"b20_24_train_metadata.csv"
        train_df.to_csv(str(train_target), index=False)

        val_target = Path(tmp_dir) / f"b20_24_val_metadata.csv"
        val_df.to_csv(str(val_target), index=False)

        mlflow.log_artifact(str(target))
        mlflow.log_artifact(str(train_target))
        mlflow.log_artifact(str(val_target))


if __name__ == "__main__":
    main()
