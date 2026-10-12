from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import mlflow
import pandas as pd
from lightning import Callback, LightningModule, Trainer


class CarcinomaPredictionTableCallbackBase(Callback, ABC):
    """Saves tile-level predictions as a table in the test or predict stage.

    In the test stage (labeled data), the tile targets are saved as well.
    Subclasses only extract tiles (slide, x, y, prediction, target) from a batch
    and pass them to `_add_rows`.
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ) -> None:
        self.table: dict[str, list[Any]] = {
            "slide": [],
            "x": [],
            "y": [],
            "prediction": [],
        }
        if stage == "test":
            self.table["target"] = []

    def _add_rows(
        self,
        slide: str,
        xs: Iterable[int],
        ys: Iterable[int],
        predictions: Iterable[float],
        targets: Iterable[bool] | None = None,
    ) -> None:
        predictions = list(predictions)
        self.table["slide"].extend([slide] * len(predictions))
        self.table["x"].extend(xs)
        self.table["y"].extend(ys)
        self.table["prediction"].extend(predictions)
        if targets is not None:
            self.table["target"].extend(targets)

    @abstractmethod
    def _on_batch_end(self, outputs: Any, batch: Any) -> None: ...

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(outputs, batch)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._on_batch_end(outputs, batch)

    def _save_table(self) -> None:
        df = pd.DataFrame(self.table)
        df.to_json("carcinoma_prediction_table.json", orient="split")
        mlflow.log_artifact(
            "carcinoma_prediction_table.json",
            artifact_path="tables",
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        self._save_table()

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_predict_epoch_end(trainer, pl_module)
        self._save_table()
