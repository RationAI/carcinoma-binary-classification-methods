"""Finds a tile threshold for a candidate model that matches a reference heatmap.

The reference model's tile predictions, binarized by an expert threshold (e.g. set
by a pathologist from validation heatmaps), are used as ground truth for the
candidate model. The candidate threshold is the one maximizing F1 (= Dice of the
binarized heatmaps), the same way as for supervised tile thresholds. Both
prediction tables must come from the same tiles (joined on slide, x, y).
"""

from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import f1_score

from postprocessing.slide_level_curves import perform_pr


KEYS = ["slide", "x", "y"]


def joined_table(
    reference_uri: str, candidate_uri: str, reference_threshold: float
) -> pd.DataFrame:
    """Candidate predictions with binarized reference predictions as target."""
    reference = pd.read_json(
        mlflow.artifacts.download_artifacts(reference_uri), orient="split"
    )
    candidate = pd.read_json(
        mlflow.artifacts.download_artifacts(candidate_uri), orient="split"
    )
    df = candidate[[*KEYS, "prediction"]].merge(
        reference[[*KEYS, "prediction"]],
        on=KEYS,
        suffixes=("", "_reference"),
        validate="1:1",
    )
    if len(df) < max(len(reference), len(candidate)):
        print(
            f"Warning: only {len(df)} tiles matched "
            f"(reference {len(reference)}, candidate {len(candidate)})"
        )

    df["target"] = df.pop("prediction_reference") > reference_threshold
    return df


@with_cli_args(["+postprocessing=tile_threshold_matching"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = joined_table(
        config.reference_preds_uri, config.preds_uri, config.reference_threshold
    )
    pr_path, pr_t = perform_pr(df, "prediction", "tile_precision_recall.png")
    f1 = f1_score(df["target"], df["prediction"] >= pr_t)

    logger.log_hyperparams(
        {"reference_threshold": config.reference_threshold, "pr_threshold": pr_t}
    )
    logger.log_metrics(
        {
            "f1": float(f1),
            "matched_tiles": float(len(df)),
            "reference_positive_fraction": float(df["target"].mean()),
        }
    )
    print(f"pr_threshold: {pr_t:.6f}, F1 (Dice) vs reference: {f1:.4f}")

    logger.log_artifact(local_path=pr_path, artifact_path="plots")
    Path(pr_path).unlink()


if __name__ == "__main__":
    main()
