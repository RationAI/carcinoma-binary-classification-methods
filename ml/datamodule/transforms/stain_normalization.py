from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from rationai.staining import (
    ColorConversion,
    NormalizeStainingTransform,
    try_estimate_stain_vectors,
)


StainTuple = tuple[float, float, float]


def _to_stain_tuple(stain: Sequence[float] | None) -> StainTuple | None:
    if stain is None:
        return None
    return cast("StainTuple", tuple(float(v) for v in stain))


class TileStainNormalizer:
    """Normalizes every tile from its own H&E stain vectors to the reference stains.

    The input stain vectors are estimated per tile. When the estimate is unreliable
    (see `try_estimate_stain_vectors`), the generic stains of the dataset are used.

    Stains are given as mappings with `hematoxylin`, `eosin` and an optional
    `residual` key (the same format as `stains` in the data configs).
    """

    def __init__(
        self,
        reference_stains: Mapping[str, Sequence[float]],
        fallback_stains: Mapping[str, Sequence[float]],
        stain_similarity_threshold: float = 8.0,
        stain_channel_correlation_threshold: float = 0.0,
        exclude_background: bool = True,
    ) -> None:
        self.target_stain1 = cast(
            "StainTuple", _to_stain_tuple(reference_stains["hematoxylin"])
        )
        self.target_stain2 = cast(
            "StainTuple", _to_stain_tuple(reference_stains["eosin"])
        )
        self.target_stain3 = _to_stain_tuple(reference_stains.get("residual"))

        self.stain_similarity_threshold = stain_similarity_threshold
        self.stain_channel_correlation_threshold = stain_channel_correlation_threshold
        self.exclude_background = exclude_background

        fallback_conversion = ColorConversion.from_stain_vectors(
            cast("StainTuple", _to_stain_tuple(fallback_stains["hematoxylin"])),
            cast("StainTuple", _to_stain_tuple(fallback_stains["eosin"])),
            _to_stain_tuple(fallback_stains.get("residual")),
        )
        self.fallback_transform = self._build_transform(fallback_conversion.matrix)

    def _build_transform(
        self, rgb2stain: NDArray[np.float64]
    ) -> NormalizeStainingTransform:
        return NormalizeStainingTransform(
            rgb2stain=rgb2stain,
            target_stain1=self.target_stain1,
            target_stain2=self.target_stain2,
            target_stain3=self.target_stain3,
            exclude_background=self.exclude_background,
        )

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        success, stain1, stain2 = try_estimate_stain_vectors(
            image,
            # no artifact masks are available on the tile level
            artifact_mask=np.zeros(image.shape[:2], dtype=bool),
            stain_similarity_threshold=self.stain_similarity_threshold,
            stain_channel_correlation_threshold=self.stain_channel_correlation_threshold,
        )

        if success:
            conversion = ColorConversion.from_stain_vectors(stain1, stain2)
            transform = self._build_transform(conversion.matrix)
        else:
            transform = self.fallback_transform

        return transform(image=image)["image"]
