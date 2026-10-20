from ml.callbacks.carcinoma_prediction_table_callback_base import (
    CarcinomaPredictionTableCallbackBase,
)
from ml.typing import (
    LabeledBagOfTilesSampleBatch,
    MILModelOutput,
    UnlabeledBagOfTilesSampleBatch,
)


class CarcinomaPredictionTableCallbackMIL(CarcinomaPredictionTableCallbackBase):
    """MIL model variant: saves TL predictions of each (padded) bag in a batch."""

    def _on_batch_end(
        self,
        outputs: MILModelOutput,
        batch: UnlabeledBagOfTilesSampleBatch | LabeledBagOfTilesSampleBatch,
    ) -> None:
        _, tl_preds, batch_mask, _ = outputs
        metadata_batch = batch[-1]

        for i, metadata in enumerate(metadata_batch):
            # remove padded tiles
            mask_slide = batch_mask[i].bool()
            preds = tl_preds[i][mask_slide]
            assert len(preds) == len(metadata["xs"]), "Tiles do not match metadata"

            targets = None
            if len(batch) == 4:
                # Test step
                targets = batch[1][i][mask_slide].bool().tolist()

            self._add_rows(
                slide=metadata["slide_name"],
                xs=metadata["xs"].tolist(),
                ys=metadata["ys"].tolist(),
                predictions=preds.tolist(),
                targets=targets,
            )
