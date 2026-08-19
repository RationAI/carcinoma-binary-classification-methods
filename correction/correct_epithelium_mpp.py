"""Epithelium Mask MPP Correction.

We have found a bug in the C20-24 `epithelium_binary` masks: the pixel content of
each mask is correctly aligned with its slide, but the embedded XResolution/
YResolution (mpp) tag is wrong, making pixels * mpp (physical size) come out to
roughly half the true slide extent. This breaks
ratiopath.tiling.overlays.tile_overlay_overlap in preprocessing/tiling_v2, which
positions/scales its read window purely from each mask's own declared mpp, so
epithelium_roi_percentage is computed against the wrong region of the mask.

For every slide, we recompute the correct level-0 mpp from first principles as
(real_slide_physical_size / epithelium_pixel_dims), re-encode the pyramid with
rationai.masks.write_big_tiff using that corrected mpp (pixel data is read from the
existing file and passed through unchanged), and log the result as a fresh artifact
in a new MLflow run.
"""

import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import pyvips
from omegaconf import DictConfig
from rationai.masks import write_big_tiff
from rationai.mlkit import with_cli_args
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


def process_one(slide_path: str, masks_dir: Path, out_dir: Path) -> None:
    stem = Path(slide_path).stem

    with OpenSlide(slide_path) as slide:
        w, h = slide.level_dimensions[0]
        mpp_x, mpp_y = slide.slide_resolution(0)
    phys_w, phys_h = w * mpp_x, h * mpp_y

    local_epi = masks_dir / f"{stem}.tiff"

    with OpenSlide(str(local_epi)) as slide:
        epi_w, epi_h = slide.level_dimensions[0]

    corrected_mpp_x = phys_w / epi_w
    corrected_mpp_y = phys_h / epi_h

    img = pyvips.Image.new_from_file(str(local_epi))
    fixed_path = out_dir / f"{stem}.tiff"
    write_big_tiff(
        img,
        fixed_path,
        corrected_mpp_x,
        corrected_mpp_y,
        tile_width=512,
        tile_height=512,
    )

    mlflow.log_artifact(str(fixed_path), artifact_path="epithelium")
    fixed_path.unlink()


@with_cli_args(["+correction=correct_epithelium_mpp"])
@hydra.main(config_path="../configs", config_name="correction", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_df = pd.read_csv(mlflow.artifacts.download_artifacts(config.slides_metadata))

    with tempfile.TemporaryDirectory(prefix="epi_fix_") as work_dir:
        masks_dir = Path(
            mlflow.artifacts.download_artifacts(
                config.epithelium_masks_uri, dst_path=work_dir
            )
        )
        out_dir = Path(work_dir) / "corrected"
        out_dir.mkdir()

        for i, slide_path in enumerate(slides_df["slide_path"], 1):
            process_one(slide_path, masks_dir, out_dir)
            print(f"[{i}/{len(slides_df)}] {Path(slide_path).stem}: fixed")


if __name__ == "__main__":
    main()
