"""These Datasets were taken from Adam Kukučka Ulcerative Colitis project and modified."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from rationai.mlkit.data.datasets.slides_tiles_loader import SlidesTilesLoader
from torch.utils.data import Dataset

from ml.datamodule.datasets.source import resolve_slides_source
from ml.typing import (
    LabeledBagOfTilesSample,
    SlideMetadata,
    SLLabeledBagOfTilesSample,
    TilingSlideMetadata,
    UnlabeledBagOfTilesSample,
)


T = TypeVar(
    "T",
    bound=LabeledBagOfTilesSample
    | SLLabeledBagOfTilesSample
    | UnlabeledBagOfTilesSample,
)


class BagOfEmbeddingsDataset(Dataset[T], ABC, Generic[T]):
    """Base for bag-of-embeddings (MIL) datasets: one item per slide.

    Handles loading slide/tile metadata, assembling the (padded) bag of tile
    embeddings and building the shared slide-level metadata. Subclasses only
    decide which labels (if any) accompany the bag.
    """

    def __init__(
        self,
        uris: Iterable[str] | None = None,
        paths: Iterable[str | Path] | None = None,
        use_paths: bool = False,
        padding: bool = True,
    ) -> None:
        self._meta = SlidesTilesLoader(
            **resolve_slides_source(uris=uris, paths=paths, use_paths=use_paths)
        )
        self.slides = self._meta.slides
        self.tiles = self._meta.tiles
        self.padding = padding

        slide_ids = self.tiles.with_format("arrow")["slide_id"]
        self.max_embeddings = pc.max(pc.value_counts(slide_ids).field("counts")).as_py()

    def __len__(self) -> int:
        return len(self.slides)

    def _load_bag(
        self, idx: int
    ) -> tuple[TilingSlideMetadata, pa.Table, torch.Tensor, SlideMetadata]:
        slide_metadata = self.slides[idx]

        slide_name = Path(slide_metadata["path"]).stem

        # materialize the slide's tiles as one Arrow table in a single query and
        # read every column from it. Never index the HF dataset by column name
        # here: `ds["x"]` returns a lazy `Column`, and `torch.tensor(column)`
        # then fetches it element by element, each fetch re-selecting the column
        # on the full (61M-row) backing table -> minutes to hours per slide
        slide_tiles = self._meta.filter_tiles_by_slide(slide_metadata["id"])
        slide_tiles = slide_tiles.with_format("arrow")[:]

        embeddings = slide_tiles["embedding"].combine_chunks()
        slide_embeddings = torch.from_numpy(
            embeddings.flatten()  # unlike .values, respects slice offsets
            .to_numpy()
            .astype(np.float32)  # copy -> writable tensor
            .reshape(len(embeddings), -1)
        )

        pad_amount = self.max_embeddings - slide_embeddings.shape[0]
        assert pad_amount >= 0, "Invalid padding"

        if self.padding:
            slide_embeddings = F.pad(
                slide_embeddings,
                (0, 0, 0, pad_amount),
                value=0.0,
            )

        metadata = SlideMetadata(
            slide_id=slide_metadata["id"],
            slide_name=slide_name,
            slide_path=slide_metadata["path"],
            xs=torch.tensor(slide_tiles["x"].to_numpy()),
            ys=torch.tensor(slide_tiles["y"].to_numpy()),
        )

        return slide_metadata, slide_tiles, slide_embeddings, metadata

    @abstractmethod
    def __getitem__(self, idx: int) -> T: ...


class UnlabeledBagOfEmbeddingsDataset(
    BagOfEmbeddingsDataset[UnlabeledBagOfTilesSample]
):
    def __init__(
        self,
        uris: Iterable[str] | None = None,
        paths: Iterable[str | Path] | None = None,
        use_paths: bool = False,
        padding: bool = True,
    ) -> None:
        super().__init__(uris=uris, paths=paths, use_paths=use_paths, padding=padding)

    def __getitem__(self, idx: int) -> UnlabeledBagOfTilesSample:
        _, _, slide_embeddings, metadata = self._load_bag(idx)
        return slide_embeddings, metadata


class SLLabeledBagOfEmbeddingsDataset(
    BagOfEmbeddingsDataset[SLLabeledBagOfTilesSample]
):
    """Bag-of-embeddings dataset carrying only slide-level (SL) labels.

    Unlike `LabeledBagOfEmbeddingsDataset`, this does not require tile-level
    (TL) carcinoma annotations, so it can be used with data that only has
    slide-level ground truth (classic MIL, no TL supervision).
    """

    def __init__(
        self,
        uris: Iterable[str] | None = None,
        paths: Iterable[str | Path] | None = None,
        use_paths: bool = False,
        padding: bool = True,
    ) -> None:
        super().__init__(uris=uris, paths=paths, use_paths=use_paths, padding=padding)

    def __getitem__(self, idx: int) -> SLLabeledBagOfTilesSample:
        slide_metadata, _, slide_embeddings, metadata = self._load_bag(idx)

        sl_label = torch.tensor(slide_metadata["carcinoma"]).float()

        return slide_embeddings, sl_label, metadata


class LabeledBagOfEmbeddingsDataset(BagOfEmbeddingsDataset[LabeledBagOfTilesSample]):
    """Bag-of-embeddings dataset carrying both SL and TL labels (hybrid MIL)."""

    def __init__(
        self,
        carcinoma_roi_t: float,
        uris: Iterable[str] | None = None,
        paths: Iterable[str | Path] | None = None,
        use_paths: bool = False,
        padding: bool = True,
    ) -> None:
        super().__init__(uris=uris, paths=paths, use_paths=use_paths, padding=padding)
        self.carcinoma_roi_t = carcinoma_roi_t

        self.slide_carcinoma = dict(
            zip(
                self.slides["id"],
                self.slides["carcinoma"],
                strict=True,
            )
        )

        def label_row(row: dict[str, Any]) -> dict[str, bool]:
            # if negative slide, all its tiles are negative
            if not self.slide_carcinoma[row["slide_id"]]:
                return {"carcinoma": False}

            # if positive slide, get the overlap (either epithelium or carcinoma)
            roi_percentage = (
                row["carcinoma_roi_percentage"]
                if "carcinoma_roi_percentage" in row
                else row["epithelium_roi_percentage"]
            )

            # and threshold it
            return {"carcinoma": roi_percentage > self.carcinoma_roi_t}

        self.tiles = self.tiles.map(label_row)
        self._meta.tiles = self.tiles
        # no need to re-build index after .map

        # note that there is no "stratified filtering" unlike in tile-level setup
        # this is due to the fact that here we need to preserve slide structure for the inference
        # not to fool the attention module -> if we represent positive slides only by the positive tiles
        # the attention aggregation mechanism might collapse in the inference where we represent positive
        # slides with all the tiles (including negative). In the tile-lvel setup this was not a problem because
        # a sample on which the model operates is tile not slide (there is no notion of slide in that setup)

    def __getitem__(self, idx: int) -> LabeledBagOfTilesSample:
        slide_metadata, slide_tiles, slide_embeddings, metadata = self._load_bag(idx)

        sl_label = torch.tensor(slide_metadata["carcinoma"]).float()

        tl_labels = torch.zeros(len(slide_embeddings)).float()  # pad with zero labels
        tl_labels[: len(slide_tiles)] = torch.tensor(
            slide_tiles["carcinoma"].to_numpy(zero_copy_only=False)
        ).float()

        return slide_embeddings, tl_labels, sl_label, metadata
