import re
from typing import Any

from torch import Tensor, nn
from transformers import ViTModel

from ml.base_model import CarcinomaTileModel


# Newer transformers renamed the ViT parameters. Some checkpoints (breast) were
# saved with the new names, so map them back to the names of the pinned version.
_NEW_TO_PINNED_VIT_KEYS = [
    (
        r"\.layers\.(\d+)\.attention\.q_proj\.",
        r".encoder.layer.\1.attention.attention.query.",
    ),
    (
        r"\.layers\.(\d+)\.attention\.k_proj\.",
        r".encoder.layer.\1.attention.attention.key.",
    ),
    (
        r"\.layers\.(\d+)\.attention\.v_proj\.",
        r".encoder.layer.\1.attention.attention.value.",
    ),
    (
        r"\.layers\.(\d+)\.attention\.o_proj\.",
        r".encoder.layer.\1.attention.output.dense.",
    ),
    (r"\.layers\.(\d+)\.mlp\.fc1\.", r".encoder.layer.\1.intermediate.dense."),
    (r"\.layers\.(\d+)\.mlp\.fc2\.", r".encoder.layer.\1.output.dense."),
    (r"\.layers\.(\d+)\.", r".encoder.layer.\1."),
]


def _to_pinned_vit_key(key: str) -> str:
    for pattern, replacement in _NEW_TO_PINNED_VIT_KEYS:
        new_key, n = re.subn(pattern, replacement, key)
        if n:
            return new_key
    return key


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

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            _to_pinned_vit_key(k) if k.startswith("backbone.") else k: v
            for k, v in checkpoint["state_dict"].items()
        }
