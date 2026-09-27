import random
from abc import ABC
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from albumentations.core.composition import TransformType
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from ml.datamodule.datasets.source import resolve_slides_source
from ml.typing import (
    LabeledTileSample,
    TilingSlideMetadata,
    UnlabeledTileSample,
)


T_co = TypeVar("T_co", covariant=True)


def get_slide_name(slide_metadata: TilingSlideMetadata) -> str:
    return Path(slide_metadata["path"]).stem


class BaseSingleSlideDataset(Dataset[LabeledTileSample | UnlabeledTileSample], ABC):
    def __init__(
        self,
        slide_metadata: TilingSlideMetadata,
        tiles: HFDataset,
        include_label: bool,
    ) -> None:
        super().__init__()
        self.include_label = include_label
        self.slide_metadata = slide_metadata
        self.tiles = tiles
        if len(tiles) == 0:
            print(
                f"Warning: No tiles found for slide {get_slide_name(slide_metadata)}."
            )


class BaseTileDataset(MetaTiledSlides[T_co]):
    """This class abstracts the functionality shared across embedding and image datasets."""

    def __init__(
        self,
        single_slide_ds_cls: type[
            BaseSingleSlideDataset
        ],  # dataset class for tiles of a single slide
        uris: Iterable[str] | None = None,  # MLFlow URI(s) of tiled dataset
        paths: Iterable[str | Path]
        | None = None,  # local directory path(s) of tiled dataset
        use_paths: bool = False,  # if both uris and paths given, which one to use
        carcinoma_roi_t: float | None = None,  # only for labeled
        stratified_filter: bool | None = None,  # only for labeled
        train_pos_tissue_roi_t: float
        | None = None,  # epithelium based training in labeled mode,
        transforms: TransformType | None = None,
        num_slides: int | None = None,  # cap slide count for very large datasets
        slide_range: tuple[int | None, int | None]
        | None = None,  # (start, end) slide index range, end inclusive; None bounds mean "from first"/"to last". Should be set only if want to select specific range of slides
    ) -> None:
        self.labeled = carcinoma_roi_t is not None and stratified_filter is not None
        self.train_pos_tissue_roi_t = train_pos_tissue_roi_t
        self.stratified_filter = stratified_filter
        self.carcinoma_roi_t = carcinoma_roi_t
        if (carcinoma_roi_t is None) ^ (stratified_filter is None):
            raise ValueError(
                "Either both should be None -> unlabeled mode, or both set -> labeled mode"
            )

        self.transforms = transforms
        self.single_slide_ds_cls = single_slide_ds_cls

        self.num_slides = num_slides
        self.slide_range = slide_range
        if num_slides is not None and slide_range is not None:
            raise ValueError(
                "Cannot use both deterministsic and non-deterministic subsampling"
            )

        super().__init__(
            **resolve_slides_source(uris=uris, paths=paths, use_paths=use_paths)
        )

    def _slide_carcinoma_map(self) -> dict[str, bool]:
        return dict(
            zip(
                self.slides["id"],
                self.slides["carcinoma"],
                strict=True,
            )
        )

    def filter_non_carcinoma(self, tiles: HFDataset) -> HFDataset:
        """Filter negative tiles from positive slides."""
        assert self.labeled, "Only allowed for labeled dataset"

        # vectorized over the small metadata columns only, so the embedding
        # column is never deserialized
        tile_columns = tiles.with_format("arrow")
        slide_is_pos = self._positive_slide_mask(tile_columns["slide_id"])

        # negative tiles in positive slides are filtered
        keep = pc.invert(pc.and_(slide_is_pos, pc.invert(tile_columns["carcinoma"])))

        # breast training specific filter:
        # filter edge tiles which may contain wrongly detected epithelium
        if self.train_pos_tissue_roi_t is not None:
            roi_ok = pc.greater_equal(
                tile_columns["tissue_roi_percentage"], self.train_pos_tissue_roi_t
            )
            keep = pc.and_(keep, pc.or_(pc.invert(slide_is_pos), roi_ok))

        keep = pc.fill_null(keep, False)
        return tiles.select(np.flatnonzero(keep.to_numpy(zero_copy_only=False)))

    def _subset_slides(
        self, slides: HFDataset, tiles: HFDataset, deterministic: bool
    ) -> tuple[HFDataset, HFDataset]:
        """Restricts slides/tiles to a uniform random or determinisitc sample of `self.num_slides`."""
        if deterministic:
            assert self.slide_range is not None
            start, end = self.slide_range
            start_idx = 0 if start is None else start
            stop_idx = len(slides) if end is None else end + 1  # end is inclusive

            if start_idx < 0 or stop_idx < 0 or stop_idx < start_idx:
                raise ValueError("Invalid bounds")

            selected_ids = set(slides["id"][start_idx:stop_idx])
        else:
            assert self.num_slides is not None
            selected_ids = set(random.sample(slides["id"], self.num_slides))

        return (
            self._select_where_in(slides, "id", selected_ids),
            self._select_where_in(tiles, "slide_id", selected_ids),
        )

    @staticmethod
    def _select_where_in(
        ds: HFDataset, column: str, values: Iterable[str]
    ) -> HFDataset:
        """Vectorized `ds.filter(row[column] in values)` reading only `column`."""
        col = ds.with_format("arrow")[column]
        mask = pc.is_in(col, value_set=pa.array(list(values)).cast(col.type))
        return ds.select(np.flatnonzero(mask.to_numpy(zero_copy_only=False)))

    def _positive_slide_mask(self, slide_ids: pa.ChunkedArray) -> pa.ChunkedArray:
        positive = [k for k, v in self._slide_carcinoma_map().items() if v]
        return pc.is_in(slide_ids, value_set=pa.array(positive).cast(slide_ids.type))

    def _print_filtering_summary(self, slides: HFDataset, tiles: HFDataset) -> None:
        """Debug print of what remains after subsetting/labeling/filtering."""
        tile_columns = tiles.with_format("arrow")
        slides_with_tiles = len(pc.unique(tile_columns["slide_id"]))
        msg = (
            f"[{type(self).__name__}] remaining: {len(slides)} slides "
            f"({slides_with_tiles} with tiles), {len(tiles)} tiles"
        )

        if self.labeled:
            slide_labels = Counter(slides["carcinoma"])
            tile_labels = Counter(
                {
                    item["values"].as_py(): item["counts"].as_py()
                    for item in pc.value_counts(tile_columns["carcinoma"])
                }
            )
            msg += (
                f" | slides: {slide_labels[True]} carcinoma / "
                f"{slide_labels[False]} non-carcinoma"
                f" | tiles: {tile_labels[True]} carcinoma / "
                f"{tile_labels[False]} non-carcinoma"
            )

        print(msg)

    def resample_slides(self) -> None:
        """Redraws a fresh random sample of `self.num_slides` slides.

        Rebuilds the underlying per-slide datasets in place (e.g. once per
        training epoch).
        """
        self.datasets = list(self.generate_datasets())
        self.cumulative_sizes = self.cumsum(self.datasets)

    def generate_datasets(self) -> Iterable[Dataset[T_co]]:
        # cache the full, unfiltered slides/tiles once so that repeated
        # (re)sampling always draws from the complete pool, not a previous subset
        if not hasattr(self, "_all_slides"):
            self._all_slides = self.slides
            self._all_tiles = self.tiles

        slides, tiles = self._all_slides, self._all_tiles

        if self.slide_range is not None:
            slides, tiles = self._subset_slides(slides, tiles, True)

        if self.num_slides is not None:
            slides, tiles = self._subset_slides(slides, tiles, False)

        # subsetting leaves an indices mapping; flatten it so
        # filter_tiles_by_slide()'s per-sample .select() stays contiguous.
        # flatten_indices() is never a no-op (it always does a full map()),
        # so skip it when no subsetting happened.
        if self.slide_range is not None or self.num_slides is not None:
            slides = slides.flatten_indices()
            tiles = tiles.flatten_indices()

        self.slides = slides

        if self.labeled:
            # negative slides are never carcinoma, regardless of tile-level overlap
            # (e.g. epithelium tiles in negative slides are not carcinoma).
            # positive slides decide per-tile via carcinoma annotation (if present)
            # or epithelium annotation (weak substitute), thresholded.
            assert self.carcinoma_roi_t is not None

            # vectorized over the small metadata columns only; .map() would
            # deserialize and rewrite the embedding column of every row
            roi_col = (
                "carcinoma_roi_percentage"
                if "carcinoma_roi_percentage" in tiles.column_names
                else "epithelium_roi_percentage"
            )
            tile_columns = tiles.with_format("arrow")
            slide_is_pos = self._positive_slide_mask(tile_columns["slide_id"])
            above_t = pc.greater(tile_columns[roi_col], self.carcinoma_roi_t)
            labels = pc.fill_null(pc.and_(slide_is_pos, above_t), False)

            if "carcinoma" in tiles.column_names:
                tiles = tiles.remove_columns("carcinoma")
            tiles = tiles.add_column("carcinoma", labels.to_pylist())

            if self.stratified_filter:
                tiles = self.filter_non_carcinoma(tiles)

        self._print_filtering_summary(slides, tiles)

        # after this, global tiles are enhanced with carcinoma, possibly
        # filtered (if labeled stratified case), and possibly subset to fewer
        # slides -- the tile index is rebuilt to match (once per call, e.g.
        # on init and on each resample_slides())
        self.tiles = tiles
        self._meta.tiles = tiles
        self._meta._slide_id_to_indices = self._meta._build_tile_index(tiles)

        return (
            cast(
                "Dataset[T_co]",
                self.single_slide_ds_cls(
                    slide,
                    tiles=self._meta.filter_tiles_by_slide(slide["id"]),
                    include_label=self.labeled,
                    **({"transforms": self.transforms} if self.transforms else {}),
                ),
            )
            for slide in self.slides
        )
