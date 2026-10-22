from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from ml.callbacks.carcinoma_prediction_table_callback_base import (
    CarcinomaPredictionTableCallbackBase,
)
from ml.typing import (
    LabeledTileSampleBatch,
    TilingSlideMetadata,
    UnlabeledTileSampleBatch,
)


if TYPE_CHECKING:
    from ml.datamodule import TileDataModule


class CarcinomaPredictionTableCallback(
    CarcinomaPredictionTableCallbackBase, MultiloaderLifecycle
):
    """Tile model variant: each dataloader corresponds to one slide."""

    def _on_dataloader_start(
        self, mode: str, trainer: Trainer, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("TileDataModule", trainer.datamodule)
        self.slide = cast(
            "TilingSlideMetadata", getattr(datamodule, mode).slides[dataloader_idx]
        )

    def on_test_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("test", trainer, dataloader_idx)

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("predict", trainer, dataloader_idx)

    def _on_batch_end(
        self,
        outputs: torch.Tensor,
        batch: UnlabeledTileSampleBatch | LabeledTileSampleBatch,
    ) -> None:
        targets = None
        if len(batch) == 3:
            # Test step
            _, labels, metadata = batch
            targets = labels.flatten().bool().tolist()
        else:
            # Predict step
            _, metadata = batch

        self._add_rows(
            slide=Path(self.slide["path"]).stem,
            xs=metadata["x"].tolist(),
            ys=metadata["y"].tolist(),
            predictions=outputs.flatten().tolist(),
            targets=targets,
        )
