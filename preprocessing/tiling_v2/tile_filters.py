from abc import ABC, abstractmethod

import pandas as pd


class TileFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        tiles: pd.DataFrame,
        slides: pd.DataFrame,
        metadata: pd.DataFrame | None,
    ) -> pd.Series:
        """Returns a boolean mask of tiles to keep.

        Args:
            tiles: Content of tiles.parquet.
            slides: Content of slides.parquet.
            metadata: Dataset metadata table (`data.metadata_table`), if available.
        """


class OverlapThresholdFilter(TileFilter):
    """Thresholds the common tissue and QC overlaps with `<mask>_t` thresholds.

    Only the `<mask>_percentage` columns of `masks` are used, all other overlaps
    (annotations, epithelium, ...) are labeling/descriptive and handled elsewhere.
    Tissue overlap has to be above the threshold, every other overlap (blur,
    folding, ...) has to be at most the threshold. A mask without a column
    or a threshold is skipped.
    """

    masks = ("tissue_roi", "residual", "blur", "folding", "exclude")

    def __init__(self, thresholds: dict[str, float]) -> None:
        self.thresholds = thresholds

    def __call__(
        self,
        tiles: pd.DataFrame,
        slides: pd.DataFrame,
        metadata: pd.DataFrame | None,
    ) -> pd.Series:
        mask = pd.Series(True, index=tiles.index)
        for name in self.masks:
            col, t = f"{name}_percentage", f"{name}_t"
            if col not in tiles.columns or t not in self.thresholds:
                continue

            mask &= (
                tiles[col] > self.thresholds[t]
                if name == "tissue_roi"
                else tiles[col] <= self.thresholds[t]
            )

        return mask


class AnnotationOverlapFilter(TileFilter):
    """Keeps tiles that sufficiently overlap at least one annotation mask.

    Each `<mask>_t` threshold is compared against the `<mask>_percentage` column.
    Optionally, all tiles of some slides are kept regardless of annotations
    (e.g. BRACS normal slides, which have no annotations). These slides are
    selected by `metadata[keep_slides_column]` being one of `keep_slides_values`.
    """

    def __init__(
        self,
        thresholds: dict[str, float],
        keep_slides_column: str | None = None,
        keep_slides_values: list[str] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.keep_slides_column = keep_slides_column
        self.keep_slides_values = list(keep_slides_values or [])

    def __call__(
        self,
        tiles: pd.DataFrame,
        slides: pd.DataFrame,
        metadata: pd.DataFrame | None,
    ) -> pd.Series:
        mask = pd.Series(False, index=tiles.index)
        for t, value in self.thresholds.items():
            mask |= tiles[t.removesuffix("_t") + "_percentage"] > value

        if self.keep_slides_column is not None:
            if metadata is None:
                raise ValueError("keep_slides_column requires data.metadata_table")

            keep = metadata[self.keep_slides_column].isin(self.keep_slides_values)
            keep_ids = slides.loc[
                slides["path"].isin(metadata.loc[keep, "slide_path"]), "id"
            ]
            mask |= tiles["slide_id"].isin(keep_ids)

        return mask


def apply_filters(
    tiles: pd.DataFrame,
    filters: dict[str, TileFilter],
    slides: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Applies all filters (logical AND) and reports how many tiles each one drops."""
    mask = pd.Series(True, index=tiles.index)
    for name, tile_filter in filters.items():
        filter_mask = tile_filter(tiles, slides, metadata)
        print(f"Filter '{name}' drops {(~filter_mask).sum()}/{len(tiles)} tiles")
        mask &= filter_mask

    print(f"Keeping {mask.sum()}/{len(tiles)} tiles")
    return tiles[mask]
