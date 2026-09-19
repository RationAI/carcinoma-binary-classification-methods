"""Verifies embeddings stored in merged (sharded) tile parquets.

For randomly chosen slides and tiles, the tile image is re-read from the slide,
re-embedded with the encoder and compared with the stored `embedding` of the
same row. To detect misalignment (embeddings shifted / permuted relative to
tiles), each recomputed embedding is also compared against the stored
embeddings of all sampled tiles of the slide: the best match must be the tile
itself.
"""

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import hydra
import numpy as np
import pyarrow.compute as pc
import torch
from datasets import Dataset as HFDataset
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.data.datasets import OpenSlideTilesDataset, SlidesTilesLoader
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.datamodule.datasets.source import resolve_slides_source


if TYPE_CHECKING:
    from ml.modeling.backbone.foundation_base import FoundationModel


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def slide_rows(tiles: HFDataset, slide_id: str) -> np.ndarray:
    """Row indices of `slide_id` in file order (no group_by, so order is exact)."""
    col = tiles.with_format("arrow")["slide_id"]
    mask = pc.equal(col, slide_id)
    return np.flatnonzero(mask.to_numpy(zero_copy_only=False))


@torch.no_grad()
def embed(
    encoder: "FoundationModel",
    images: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    # same normalization as in tile_embeddings.py
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    x = torch.stack(
        [torch.from_numpy(norm(image=im)["image"]).permute(2, 0, 1) for im in images]
    )
    out = [
        encoder(x[i : i + batch_size].to(device)).float().cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out).numpy()


@with_cli_args(["+preprocessing=verify_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    encoder = encoder.to(device).eval()

    loader = SlidesTilesLoader(
        paths=[config.data.tiles_filtered_w_virchow2_path_224]
    )
    slides, tiles = loader.slides, loader.tiles
    assert "embedding" in tiles.column_names, "Tiles have no `embedding` column"

    chosen = random.sample(range(len(slides)), min(config.num_slides, len(slides)))
    failed = 0
    for slide_idx in chosen:
        slide = slides[slide_idx]
        slide_name = Path(slide["path"]).stem
        rows = slide_rows(tiles, slide["id"])
        if len(rows) == 0:
            print(f"[skip] {slide_name}: no tiles")
            continue

        picked = np.sort(
            np.array(random.sample(list(rows), min(config.tiles_per_slide, len(rows))))
        )
        sub = tiles.select(picked)

        reader = OpenSlideTilesDataset(
            slide_path=slide["path"],
            level=slide["level"],
            tile_extent_x=slide["tile_extent_x"],
            tile_extent_y=slide["tile_extent_y"],
            tiles=sub,
        )
        images = [reader[i] for i in range(len(reader))]
        recomputed = embed(encoder, images, device, config.batch_size)
        stored = np.asarray(sub["embedding"], dtype=np.float32)

        sim = cosine(recomputed, stored)  # sim[i, j]: recomputed i vs stored j
        self_sim = np.diag(sim)
        best = sim.argmax(1)
        mismatched = best != np.arange(len(best))
        max_abs_diff = np.abs(recomputed - stored).max(1)

        ok = self_sim.min() >= config.cosine_threshold and not mismatched.any()
        failed += not ok
        print(
            f"[{'OK' if ok else 'FAIL'}] {slide_name}: {len(picked)}/{len(rows)} tiles"
            f" | cos(self) min={self_sim.min():.4f} mean={self_sim.mean():.4f}"
            f" | best-match==self {1 - mismatched.mean():.0%}"
            f" | max|diff| median={np.median(max_abs_diff):.4f}"
        )
        if mismatched.any():
            # a consistent row offset suggests a shift / permutation of the
            # stored embeddings relative to the tiles
            offsets = (picked[best] - picked)[mismatched]
            print(
                f"       {mismatched.sum()} tiles matched another tile; "
                f"row offsets (best - self): {offsets[:10].tolist()}"
            )

    logger.log_metrics(
        {"slides_checked": len(chosen), "slides_failed": failed}
    )
    print(f"{len(chosen) - failed}/{len(chosen)} slides passed")
    if failed:
        raise RuntimeError(f"Embedding verification failed for {failed} slide(s)")


if __name__ == "__main__":
    main()
