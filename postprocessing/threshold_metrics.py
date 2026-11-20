"""Fills thresholded metrics into test runs evaluated without thresholds.

A test suite run with `null` model thresholds (see `ml.metrics.binary_metrics`)
logs the predictions, targets and AUC/AUPRC only. Once the thresholds are chosen
(set in the model checkpoint config, pulled in as `model_data`), this script
computes the rest from the run's tables and logs it into the same run, under the
same names the test suite would have used:

- TL: `tables/carcinoma_prediction_table.json` -> `test/<metric>` (tile model) or
  `tl_test/<metric>` (MIL model), plus the per-slide columns of `nested_metrics.json`.
- SL aggregators (tile model): `tables/sl_predictions_{max,mean_pool_max,top_k}.json`
  -> `prediction_binary` column and `sl/<aggregator>/<metric>`. With all three
  thresholds also `tables/sl_predictions_majority_vote.json` and
  `sl/majority_vote/<metric>`.
- SL MIL: `tables/sl_predictions.json` -> `sl_prediction_binary` column and
  `sl_test/<metric>`.

Tables are re-uploaded with the binary predictions filled, and the thresholds
used are logged as `thresholds.json`. Steps whose threshold is null are skipped.
So are steps whose metrics the run already has (e.g. SL thresholds were known at
test time, only the tile one was null), unless `overwrite=true` (e.g. after
changing a threshold).
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.entities import Metric
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args

from ml._mlflow_compat import apply_mlflow_compat_patch
from ml.callbacks.multi_aggregator_eval_callback import compute_sl_metrics
from ml.metrics import thresholded_metrics
from postprocessing.add_auprc import tile_table


def metric_values(
    preds: pd.Series, target: pd.Series, threshold: float
) -> dict[str, float]:
    preds_t = torch.tensor(preds.astype(float).values)
    target_t = torch.tensor(target.astype(int).values)
    return {
        name: metric(preds_t, target_t).item()
        for name, metric in thresholded_metrics(threshold).items()
    }


def nested_metrics(
    tiles: pd.DataFrame, nested: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Sets the per-slide thresholded columns (after AUC/AUPRC, as the callback orders them)."""
    per_slide = (
        tiles.groupby("slide")
        .apply(
            lambda g: pd.Series(metric_values(g["prediction"], g["target"], threshold)),
            include_groups=False,
        )
        .reindex(nested["key"])
    )
    if per_slide.isna().any().any():
        raise ValueError("Some nested metrics keys have no tile predictions")

    nested = nested.copy()
    for column in per_slide.columns:
        nested[column] = per_slide[column].to_numpy()
    return nested


class RunFiller:
    """Collects metrics and tables of one run, then writes them all at once."""

    def __init__(self, client: MlflowClient, run_id: str, overwrite: bool) -> None:
        self.client = client
        self.run_id = run_id
        self.overwrite = overwrite
        self.logged = client.get_run(run_id).data.metrics.keys()
        self.metrics: dict[str, float] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.artifacts = {
            a.path
            for directory in (None, "tables")
            for a in client.list_artifacts(run_id, directory)
        }

    def add(
        self, step: str, metrics: dict[str, float], tables: dict[str, pd.DataFrame]
    ) -> None:
        """Adds one step's metrics and tables, unless the run already has them."""
        if not self.overwrite and metrics.keys() & self.logged:
            print(f"Skipping {step}: metrics already logged")
            return
        self.metrics |= metrics
        self.tables |= tables

    def read(self, path: str) -> pd.DataFrame:
        return pd.read_json(
            mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path=path),
            orient="split",
        )

    def upload(self, thresholds: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for artifact_path, df in self.tables.items():
                path = Path(tmpdir, Path(artifact_path).name)
                df.to_json(path, orient="split")
                # artifact root must be passed as None (mlflow rejects ".")
                artifact_dir = str(Path(artifact_path).parent)
                self.client.log_artifact(
                    self.run_id,
                    str(path),
                    None if artifact_dir == "." else artifact_dir,
                )
                print(f"Updated {artifact_path}")

            path = Path(tmpdir, "thresholds.json")
            path.write_text(json.dumps(thresholds, indent=2))
            self.client.log_artifact(self.run_id, str(path))

        run = self.client.get_run(self.run_id)
        timestamp = run.info.end_time or int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[Metric(k, v, timestamp, 0) for k, v in self.metrics.items()],
        )


def fill_run(client: MlflowClient, run_id: str, config: DictConfig) -> None:
    print(f"--- {run_id}")
    thresholds = config.thresholds
    run = RunFiller(client, run_id, config.overwrite)
    is_mil = config.mil_sl_table in run.artifacts

    # TL + nested
    if thresholds.tile is not None and config.tl_table in run.artifacts:
        tiles = tile_table(run_id, config.tl_table)
        prefix = "tl_test/" if is_mil else "test/"
        tables = {}
        if config.nested_metrics in run.artifacts:
            tables[config.nested_metrics] = nested_metrics(
                tiles, run.read(config.nested_metrics), thresholds.tile
            )
        run.add(
            "TL",
            {
                f"{prefix}{k}": v
                for k, v in metric_values(
                    tiles["prediction"], tiles["target"], thresholds.tile
                ).items()
            },
            tables,
        )

    # SL MIL
    if is_mil and thresholds.slide is not None:
        df = run.read(config.mil_sl_table)
        df["sl_prediction_binary"] = df["sl_prediction"] >= thresholds.slide
        run.add(
            "SL MIL",
            {
                f"sl_test/{k}": v
                for k, v in metric_values(
                    df["sl_prediction"], df["target"], thresholds.slide
                ).items()
            },
            {config.mil_sl_table: df},
        )

    # SL aggregators (+ majority vote)
    votes: list[pd.DataFrame] = []
    for name, threshold in thresholds.aggregators.items():
        path = f"{config.sl_tables_prefix}{name}.json"
        if threshold is None or path not in run.artifacts:
            continue
        df = run.read(path)
        df["prediction_binary"] = df["prediction"] >= threshold
        run.add(
            f"SL {name}",
            compute_sl_metrics(
                name, df["target"].tolist(), df["prediction_binary"].tolist()
            ),
            {path: df},
        )
        votes.append(df.set_index("slide_name")[["prediction_binary", "target"]])

    if len(votes) == 3:
        majority = pd.DataFrame(
            {
                "prediction": pd.concat(
                    [v["prediction_binary"] for v in votes], axis=1
                ).sum(axis=1)
                >= 2,
                "target": votes[0]["target"],
            }
        ).reset_index()
        run.add(
            "SL majority vote",
            compute_sl_metrics(
                "majority_vote",
                majority["target"].tolist(),
                majority["prediction"].tolist(),
            ),
            {f"{config.sl_tables_prefix}majority_vote.json": majority},
        )

    if not run.metrics:
        print("Nothing to fill")
        return

    for name, value in sorted(run.metrics.items()):
        print(f"{name}: {value:.4f}")
    run.upload(OmegaConf.to_container(thresholds, resolve=True))  # type: ignore[arg-type]


@with_cli_args(["+postprocessing=threshold_metrics"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    apply_mlflow_compat_patch()
    client = MlflowClient()
    for run_id in config.run_ids:
        fill_run(client, run_id, config)


if __name__ == "__main__":
    main()
