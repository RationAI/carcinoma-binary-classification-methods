import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import albumentations as A
import hydra
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.datamodule.datasets import UnlabeledTilesDataset


if TYPE_CHECKING:
    from ml.modeling.backbone.foundation_base import FoundationModel


@with_cli_args(["+preprocessing=tile_embeddings"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    login(token=os.environ["HF_TOKEN"])
    dest = Path(config.output_path)
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)

    tiling_uri = config.data.tiles_filtered_uri_224
    slide_range = (
        None
        if config.start is None and config.end is None
        else (config.start, config.end)
    )

    with torch.no_grad():
        dataset = UnlabeledTilesDataset(
            uris=(tiling_uri,),
            transforms=A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]  # Both PGP and Wirchow2 use the same normalization. This is also a default for Albumentation.
            ),
            slide_range=slide_range,
        )

        for slide_dataset in tqdm(dataset.datasets):
            slide_name = Path(slide_dataset.slide_tiles.slide_path).stem
            out_path = (dest / slide_name).with_suffix(".pt")

            tiles = slide_dataset.slide_tiles.tiles
            xs, ys = tiles["x"], tiles["y"]  # in the order tiles are embedded

            if out_path.exists():
                try:
                    existing = torch.load(out_path, map_location="cpu")
                    # old format (bare tensor) has no coordinates -> reprocess
                    if (
                        isinstance(existing, dict)
                        and existing["embedding"].size(0) == len(slide_dataset)
                        and existing["x"].tolist() == xs
                        and existing["y"].tolist() == ys
                    ):
                        continue
                except Exception as e:  # noqa: BLE001
                    print(
                        f"{e} occured while checking existing {slide_name}, reprocessing"
                    )

            try:
                slide_dataloader = DataLoader(
                    slide_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,  # order must match the stored (x, y)
                    num_workers=config.num_workers,
                    pin_memory=device.type == "cuda",
                )
                slide_embeddings = torch.zeros(
                    (len(slide_dataset), tile_encoder.embed_dim),
                    device=device,
                    dtype=torch.float32,
                )
                batch_xs: list[torch.Tensor] = []
                batch_ys: list[torch.Tensor] = []
                for i, (x, metadata) in enumerate(slide_dataloader):
                    batch_xs.append(metadata["x"])
                    batch_ys.append(metadata["y"])
                    x = x.to(device)
                    embeddings = cast(
                        "torch.Tensor", tile_encoder(x)
                    )  # (batch_size, embed_dim)

                    start = i * config.batch_size
                    end = start + embeddings.size(0)
                    slide_embeddings[start:end] = embeddings

                # Embeddings are stored together with the coordinates of the tiles
                # they were computed for (taken from the loaded batches), so that
                # they can be matched to tiles by (x, y) and not by position.
                embedded_xs = torch.cat(batch_xs).to(torch.int64)
                embedded_ys = torch.cat(batch_ys).to(torch.int64)
                if embedded_xs.tolist() != xs or embedded_ys.tolist() != ys:
                    raise RuntimeError(
                        "Loaded tile order differs from the dataset's tile order"
                    )

                torch.save(
                    {
                        "x": embedded_xs,
                        "y": embedded_ys,
                        "embedding": slide_embeddings.cpu(),
                    },
                    out_path,
                )

            except Exception as e:  # noqa: BLE001
                print(f"{e} occured during processing {slide_name}")

    logger.log_artifacts(local_dir=config.output_path)


if __name__ == "__main__":
    main()
