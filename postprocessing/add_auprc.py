"""Adds AUPRC metrics to an already evaluated (test) run from its prediction tables.

TL: `tables/carcinoma_prediction_table.json` (tile or MIL model). If it was saved
without targets (older runs of tile models), tile labels are reconstructed by
instantiating the run's test dataset from its logged `configs/config-resolved.yaml`
(same labeling as during the evaluation) and joined on (slide, x, y). Logged as
`test/AUPRC` (tile model) or `tl_test/AUPRC` (MIL model).

SL: every `tables/sl_predictions_{name}.json` with continuous predictions is used.
Logged as `sl/{name}/AUPRC`.

Nested: per-slide AUPRC column is added to `nested_metrics.json` (computed from the
same joined TL table) and the artifact is re-uploaded.

Metrics/columns already present in the run are skipped unless `overwrite=true`.
"""

import tempfile
import time
from pathlib import Path
from typing import cast

import hydra
import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.entities import Metric
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from torchmetrics.functional.classification import binary_average_precision

from ml._mlflow_compat import apply_mlflow_compat_patch
from ml.hf_cache import configure_hf_cache
from postprocessing.read_table import read_json_table


def auprc(preds: pd.Series, target: pd.Series) -> float:
    return binary_average_precision(
        torch.tensor(preds.astype(float).values),
        torch.tensor(target.astype(int).values),
    ).item()


def tile_targets(run_id: str) -> pd.DataFrame:
    """Tile labels of the run's test dataset as (slide, x, y, target) table."""
    config_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="configs/config-resolved.yaml"
    )
    run_config = cast("DictConfig", OmegaConf.load(config_path))
    configure_hf_cache(run_config.get("hf_cache"))

    dataset_config = run_config.datamodule.test
    if "transforms" in dataset_config:
        dataset_config.transforms = None  # images are never loaded

    dataset = hydra.utils.instantiate(dataset_config)
    slide_names = {
        slide_id: Path(path).stem
        for slide_id, path in zip(
            dataset.slides["id"], dataset.slides["path"], strict=True
        )
    }

    tiles = dataset.tiles.select_columns(["slide_id", "x", "y", "carcinoma"])
    df = tiles.to_pandas()
    df["slide"] = df.pop("slide_id").map(slide_names)
    return df.rename(columns={"carcinoma": "target"})


def tile_table(run_id: str, table_path: str) -> pd.DataFrame:
    """Tile predictions joined with their targets."""
    preds = read_json_table(f"runs:/{run_id}/{table_path}")
    if "target" in preds.columns:
        return preds

    # older tables were saved without targets
    targets = tile_targets(run_id)

    df = preds.merge(targets, on=["slide", "x", "y"], how="left", validate="1:1")
    missing = df["target"].isna().sum()
    if missing:
        raise ValueError(f"{missing} predicted tiles have no target in test dataset")
    return df


def nested_auprc(tiles: pd.DataFrame, nested: pd.DataFrame) -> pd.DataFrame:
    """Adds per-slide AUPRC column (right after AUC, as the callback orders it)."""
    per_slide = tiles.groupby("slide").apply(
        lambda g: auprc(g["prediction"], g["target"]), include_groups=False
    )
    nested = nested.copy()
    nested.insert(
        nested.columns.get_loc("AUC") + 1, "AUPRC", nested["key"].map(per_slide)
    )
    if nested["AUPRC"].isna().any():
        raise ValueError("Some nested metrics keys have no tile predictions")
    return nested


def sl_auprcs(run_id: str, prefix: str) -> dict[str, float]:
    results = {}
    artifact_dir = str(Path(prefix).parent)
    for artifact in MlflowClient().list_artifacts(run_id, artifact_dir):
        if not artifact.path.startswith(prefix) or not artifact.path.endswith(".json"):
            continue

        name = artifact.path.removeprefix(prefix).removesuffix(".json")
        df = read_json_table(f"runs:/{run_id}/{artifact.path}")

        # e.g. majority vote stores only binary predictions -> AUPRC undefined
        if df["prediction"].dtype == bool:
            print(f"Skipping {artifact.path}: binary predictions only")
            continue

        results[f"sl/{name}/AUPRC"] = auprc(df["prediction"], df["target"])

    return results


@with_cli_args(["+postprocessing=add_auprc"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    apply_mlflow_compat_patch()
    client = MlflowClient()
    run = client.get_run(config.run_id)

    def done(metric: str) -> bool:
        if metric in run.data.metrics and not config.overwrite:
            print(f"Skipping {metric}: already logged")
            return True
        return False

    artifacts = {
        a.path
        for directory in (None, str(Path(config.tl_table).parent))
        for a in client.list_artifacts(config.run_id, directory)
    }

    metrics = {
        k: v
        for k, v in sl_auprcs(config.run_id, config.sl_tables_prefix).items()
        if not done(k)
    }

    nested = None
    if config.nested_metrics in artifacts:
        nested = read_json_table(f"runs:/{config.run_id}/{config.nested_metrics}")
        if "AUPRC" in nested.columns and not config.overwrite:
            print(f"Skipping {config.nested_metrics}: already has AUPRC")
            nested = None
        else:
            nested = nested.drop(columns="AUPRC", errors="ignore")

    # MIL model logs TL metrics with "tl_test/" prefix
    tl_metric = "tl_test/AUPRC" if "tl_test/AUC" in run.data.metrics else "test/AUPRC"
    if config.tl_table not in artifacts:
        print(f"Skipping TL and nested: {config.tl_table} not found in run artifacts")
    elif not done(tl_metric) or nested is not None:
        tiles = tile_table(config.run_id, config.tl_table)
        if not done(tl_metric):
            metrics[tl_metric] = auprc(tiles["prediction"], tiles["target"])
        if nested is not None:
            nested = nested_auprc(tiles, nested)
            with tempfile.TemporaryDirectory() as tmpdir:
                nested_path = Path(config.nested_metrics)
                path = Path(tmpdir, nested_path.name)
                nested.to_json(path, orient="split")
                # artifact root must be passed as None (mlflow rejects ".")
                artifact_dir = str(nested_path.parent)
                client.log_artifact(
                    config.run_id,
                    str(path),
                    None if artifact_dir == "." else artifact_dir,
                )
            print(f"Updated {config.nested_metrics} with AUPRC column")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if metrics:
        timestamp = run.info.end_time or int(time.time() * 1000)
        client.log_batch(
            config.run_id,
            metrics=[Metric(k, v, timestamp, 0) for k, v in metrics.items()],
        )


if __name__ == "__main__":
    main()
