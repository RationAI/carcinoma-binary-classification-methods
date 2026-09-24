"""Script to downsample whole slide images (WSIs) by a fixed factor from level 0."""

from pathlib import Path
from typing import cast

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote
def process_slide(
    slide_path: Path, output_path: Path, downsample_factor: float
) -> None:
    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=0)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=0))

    if float(downsample_factor).is_integer():
        # Exact box-filter averaging of factor x factor pixel blocks
        resized = slide.shrink(downsample_factor, downsample_factor)
    else:
        resized = slide.resize(1 / downsample_factor)

    resized_path = output_path / slide_path.with_suffix(".tiff").name
    write_big_tiff(
        resized,
        path=resized_path,
        mpp_x=mpp_x * downsample_factor,
        mpp_y=mpp_y * downsample_factor,
    )
    print(
        f"Processed slide {slide_path.name}: "
        f"mpp ({mpp_x:.4f}, {mpp_y:.4f}) -> "
        f"({mpp_x * downsample_factor:.4f}, {mpp_y * downsample_factor:.4f})"
    )


@with_cli_args(["+preprocessing=resize_wsi"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert config.downsample_factor > 0, "downsample_factor must be positive"

    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.data.metadata_table))
    slides_path = [Path(path) for path in df["slide_path"]]

    process_items(
        slides_path,
        fn_kwargs={
            "output_path": output_path,
            "downsample_factor": config.downsample_factor,
        },
        process_item=process_slide,
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(
        config.output_path, artifact_path=f"resized_x{config.downsample_factor}"
    )


if __name__ == "__main__":
    main()
