from lightning import LightningModule, Trainer

from ml.callbacks.curves_callback_base import CurvesCallbackBase
from ml.typing import LabeledBagOfTilesSampleBatch, MILModelOutput


class CurvesCallbackMIL(CurvesCallbackBase):
    def __init__(self, threshold: float, optimal_seek: bool) -> None:
        super().__init__(
            threshold=threshold, tile_level=True, optimal_seek=optimal_seek
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,  # type: ignore[override]
        batch: LabeledBagOfTilesSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        _, tl_outputs_raw, mask, _ = outputs
        tl_outputs_valid = tl_outputs_raw[mask.bool()]
        targets = batch[1][mask.bool()]
        self.preds.append(tl_outputs_valid.cpu())
        self.targets.append(targets.cpu())
