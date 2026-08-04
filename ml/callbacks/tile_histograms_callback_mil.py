from lightning import LightningModule, Trainer

from ml.callbacks.histograms_callback_base import (
    HistogramsCallbackBase,
)
from ml.typing import LabeledBagOfTilesSampleBatch, MILModelOutput


class TileHistogramsCallbackMIL(HistogramsCallbackBase):
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MILModelOutput,  # type: ignore[override]
        batch: LabeledBagOfTilesSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, y, _, _ = batch
        _, tl_outputs_raw, mask, _ = outputs

        tl_outputs_valid = tl_outputs_raw[mask.bool()]
        preds = tl_outputs_valid.detach().cpu().numpy().flatten()

        labels_valid = y[mask.bool()]
        labels = labels_valid.detach().cpu().numpy().flatten()

        self.all_preds.append(preds)
        self.all_labels.append(labels)
