from typing import cast

from rationai.staining import ColorConversion, NormalizeStainingTransform


def build_normalize_staining_transform(
    stain1: tuple[float, float, float],
    stain2: tuple[float, float, float],
    stain3: tuple[float, float, float],
    target_stain1: tuple[float, float, float],
    target_stain2: tuple[float, float, float],
    target_stain3: tuple[float, float, float],
) -> NormalizeStainingTransform:
    conversion = ColorConversion.from_stain_vectors(
        cast("tuple[float, float, float]", tuple(stain1)),
        cast("tuple[float, float, float]", tuple(stain2)),
        cast("tuple[float, float, float]", tuple(stain3)),
    )
    return NormalizeStainingTransform(
        rgb2stain=conversion.matrix,
        target_stain1=cast("tuple[float, float, float]", tuple(target_stain1)),
        target_stain2=cast("tuple[float, float, float]", tuple(target_stain2)),
        target_stain3=cast("tuple[float, float, float]", tuple(target_stain3)),
    )
