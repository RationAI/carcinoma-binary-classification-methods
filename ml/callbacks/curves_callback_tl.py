from typing import Any

from lightning import LightningModule, Trainer

from ml.callbacks.curves_callback_base import CurvesCallbackBase
from ml.typing import LabeledTileSampleBatch


class CurvesCallbackTile(CurvesCallbackBase):
    def __init__(self, threshold: float, optimal_seek: bool) -> None:
        super().__init__(
            threshold=threshold, tile_level=True, optimal_seek=optimal_seek
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: LabeledTileSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        targets = batch[1]
        self.preds.append(outputs.cpu())
        self.targets.append(targets.cpu())
