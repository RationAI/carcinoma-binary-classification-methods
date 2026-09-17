"""Script for copying MLFlow stored embeddings to local storage. It exists co that the long download can be executed as a job."""

import hydra
import mlflow
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+preprocessing=store_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    mlflow.artifacts.download_artifacts(
        artifact_uri=config.embeddings_uri, dst_path=config.target_path
    )


if __name__ == "__main__":
    main()
