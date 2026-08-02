from torch import Tensor

from ml.base_model import CarcinomaTileModel
from ml.modeling.decode_head import BinaryEmbeddingClassifier


class EmbeddingCarcinomaModel(CarcinomaTileModel):
    def __init__(
        self, decode_head: BinaryEmbeddingClassifier, lr: float, tl_threshold: float
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        logits = self.decode_head(x)
        return logits
