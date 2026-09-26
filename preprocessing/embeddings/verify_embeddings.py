"""Verifies embeddings stored in merged (sharded) tile parquets.

Run this before using a merged dataset for the first time. It also loads the
dataset the same way training does, so the Hugging Face datasets cache
(`HF_DATASETS_CACHE`) is built here once and reused afterwards -- point it to a
persistent location, and load from the local path (not an MLflow URI), otherwise
the cache is not reused.

For randomly chosen slides and tiles, the tile is loaded through the same
dataset class and transforms as in tile_embeddings.py, re-embedded with the
encoder and compared with the tile's stored `embedding`.
"""

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import datasets.config
import hydra
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.datamodule.datasets import UnlabeledTilesDataset


if TYPE_CHECKING:
    from ml.modeling.backbone.foundation_base import FoundationModel


@with_cli_args(["+preprocessing=verify_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    encoder = encoder.to(device).eval()

    print(f"HF datasets cache: {datasets.config.HF_DATASETS_CACHE}")
    # loading from a local path (not URI) keeps the cache key stable between jobs
    dataset = UnlabeledTilesDataset(
        paths=(config.data.tiles_filtered_w_virchow2_path_224,),
        use_paths=True,
        transforms=A.Compose(
            # must be the same as in tile_embeddings.py
            [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        ),
        num_slides=config.num_slides,  # randomly chosen (seeded above)
    )

    failed = 0
    with torch.no_grad():
        for slide_ds in dataset.datasets:
            slide_name = Path(slide_ds.slide_tiles.slide_path).stem
            tiles = slide_ds.slide_tiles.tiles
            assert "embedding" in tiles.column_names, "Tiles have no `embedding` column"

            # `slide_ds[i]` and `tiles[i]` refer to the same row
            pick = random.sample(
                range(len(slide_ds)), min(config.tiles_per_slide, len(slide_ds))
            )
            images, metadata = zip(*(slide_ds[i] for i in pick), strict=True)
            recomputed = encoder(torch.stack(images).to(device)).float().cpu()
            stored = torch.tensor(tiles.select(pick)["embedding"], dtype=torch.float32)

            l2_error = (recomputed - stored).norm(dim=1)  # (n_tiles,)
            ok = bool(l2_error.max() <= config.max_error)
            failed += not ok
            worst = int(l2_error.argmax())
            print(
                f"[{'OK' if ok else 'FAIL'}] {slide_name}: {len(pick)} tiles"
                f" | L2 error max={l2_error.max():.2e} mean={l2_error.mean():.2e}"
                f" | worst tile (x={metadata[worst]['x']}, y={metadata[worst]['y']})"
            )

    logger.log_metrics(
        {"slides_checked": len(dataset.datasets), "slides_failed": failed}
    )
    print(f"{len(dataset.datasets) - failed}/{len(dataset.datasets)} slides passed")
    if failed:
        raise RuntimeError(f"Embedding verification failed for {failed} slide(s)")


if __name__ == "__main__":
    main()
