from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import OpenSlideTilesDataset

from ml.datamodule.datasets.base import (
    BaseSingleSlideDataset,
    BaseTileDataset,
)
from ml.datamodule.transforms.stain_normalization import TileStainNormalizer
from ml.typing import (
    LabeledTileSample,
    TileMetadata,
    TilingSlideMetadata,
    UnlabeledTileSample,
)


T_co = TypeVar("T_co", covariant=True)


class TilesDataset(BaseTileDataset[T_co]):
    def __init__(
        self,
        uris: Iterable[str] | None = None,
        paths: Iterable[str | Path] | None = None,
        use_paths: bool = False,
        carcinoma_roi_t: float | None = None,
        stratified_filter: bool | None = None,
        train_pos_tissue_roi_t: float | None = None,
        transforms: TransformType | None = None,
        use_tile_stain_normalization: bool = False,
        reference_stains: Mapping[str, Sequence[float]] | None = None,
        fallback_stains: Mapping[str, Sequence[float]] | None = None,
        num_slides: int | None = None,
        slide_range: tuple[int | None, int | None] | None = None,
    ) -> None:
        self.transforms = transforms
        self.stain_normalizer: TileStainNormalizer | None = None
        if use_tile_stain_normalization:
            if reference_stains is None or fallback_stains is None:
                raise ValueError(
                    "Tile stain normalization requires both reference and fallback stains"
                )
            self.stain_normalizer = TileStainNormalizer(
                reference_stains=reference_stains, fallback_stains=fallback_stains
            )
        super().__init__(
            uris=uris,
            paths=paths,
            use_paths=use_paths,
            single_slide_ds_cls=SlideTiles,
            carcinoma_roi_t=carcinoma_roi_t,
            train_pos_tissue_roi_t=train_pos_tissue_roi_t,
            stratified_filter=stratified_filter,
            transforms=transforms,
            num_slides=num_slides,
            slide_range=slide_range,
        )

    def _single_slide_ds_kwargs(self) -> dict[str, Any]:
        return {
            **super()._single_slide_ds_kwargs(),
            "stain_normalizer": self.stain_normalizer,
        }


class LabeledTilesDataset(TilesDataset[LabeledTileSample]): ...


class UnlabeledTilesDataset(TilesDataset[UnlabeledTileSample]): ...


class SlideTiles(BaseSingleSlideDataset):
    def __init__(
        self,
        slide_metadata: TilingSlideMetadata,
        tiles: HFDataset,
        include_label: bool,
        transforms: TransformType | None = None,
        stain_normalizer: TileStainNormalizer | None = None,
    ) -> None:
        super().__init__(
            slide_metadata=slide_metadata,
            tiles=tiles,
            include_label=include_label,
        )
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata["path"],
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
        )
        self.transforms = transforms
        self.stain_normalizer = stain_normalizer
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> LabeledTileSample | UnlabeledTileSample:
        image = self.slide_tiles[idx]

        tile_row = self.slide_tiles.tiles[idx]

        metadata = TileMetadata(
            slide=self.slide_tiles.slide_path.stem,
            x=tile_row["x"],
            y=tile_row["y"],
        )

        # normalization runs on raw tiles, before any augmentation
        if self.stain_normalizer is not None:
            image = self.stain_normalizer(image)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        tensor_image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label = torch.tensor([tile_row["carcinoma"]]).float()
            return tensor_image, label, metadata

        return tensor_image, metadata
