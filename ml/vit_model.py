from torch import Tensor, nn
from transformers import ViTModel

from ml.base_model import CarcinomaTileModel


class ViTCarcinomaModel(CarcinomaTileModel):
    def __init__(
        self, backbone: ViTModel, decode_head: nn.Module, lr: float, tl_threshold: float
    ) -> None:
        super().__init__(lr=lr, tl_threshold=tl_threshold)
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x).last_hidden_state
        logits = self.decode_head(features)
        return logits
