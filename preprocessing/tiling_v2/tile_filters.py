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
    """Thresholds the `*_percentage` overlap columns with `*_t` thresholds.

    Tissue overlap has to be above the threshold, every other overlap (blur,
    folding, ...) has to be at most the threshold. Labeling and descriptive
    overlaps are not used for filtering.
    """

    ignored = ("carcinoma", "epithelium", "mucosa", "benign")

    def __init__(self, thresholds: dict[str, float]) -> None:
        self.thresholds = thresholds

    def __call__(
        self,
        tiles: pd.DataFrame,
        slides: pd.DataFrame,
        metadata: pd.DataFrame | None,
    ) -> pd.Series:
        mask = pd.Series(True, index=tiles.index)
        for col in tiles.columns:
            if not col.endswith("percentage") or any(i in col for i in self.ignored):
                continue

            t = col.replace("percentage", "t")
            if t not in self.thresholds:
                print(f"{t} for {col}")
                continue

            mask &= (
                tiles[col] > self.thresholds[t]
                if "tissue" in col
                else tiles[col] <= self.thresholds[t]
            )

        return mask


class BRACSTLFilter(TileFilter):
    """BRACS TL filtering.

    All tiles of normal slides (N) are kept.
    From the other slides, only tiles sufficiently overlapping the
    carcinoma or benign annotations are kept.
    """

    def __init__(
        self,
        carcinoma_roi_t: float,
        benign_roi_t: float,
    ) -> None:
        self.carcinoma_roi_t = carcinoma_roi_t
        self.benign_roi_t = benign_roi_t

    def __call__(
        self,
        tiles: pd.DataFrame,
        slides: pd.DataFrame,
        metadata: pd.DataFrame | None,
    ) -> pd.Series:
        if metadata is None:
            raise ValueError("BRACSTLFilter requires data.metadata_table")

        normal_paths = metadata.loc[
            metadata["WSI label"] == "N", "slide_path"
        ]
        normal_ids = slides.loc[slides["path"].isin(normal_paths), "id"]
        is_normal = tiles["slide_id"].isin(normal_ids)

        annotated = (tiles["carcinoma_roi_percentage"] > self.carcinoma_roi_t) | (
            tiles["benign_roi_percentage"] > self.benign_roi_t
        )

        return is_normal | annotated


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
