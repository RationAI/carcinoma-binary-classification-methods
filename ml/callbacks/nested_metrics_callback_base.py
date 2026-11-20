from abc import ABC
from typing import cast

import mlflow
import pandas as pd
from lightning import Callback, LightningModule, Trainer
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.metrics import NestedMetricCollection
from torchmetrics import Metric

from ml.metrics import binary_metrics


class NestedMetricsCallbackBase(Callback, ABC):
    """Calculates metrics using the `NestedMetricCollection` in the test stage, grouped per slide."""

    def __init__(self, threshold: float | None) -> None:
        super().__init__()
        # threshold None -> AUC/AUPRC only (see `ml.metrics.binary_metrics`)
        self.nested_test_metrics = NestedMetricCollection(
            metrics=cast("dict[str, Metric]", binary_metrics(threshold))
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        metrics = self.nested_test_metrics.compute()
        pd.DataFrame(metrics).to_json("nested_metrics.json", orient="split")
        mlflow.log_artifact("nested_metrics.json")
        self.nested_test_metrics.reset()
