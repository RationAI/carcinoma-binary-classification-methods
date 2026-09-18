"""Script to generate annotation masks from XML/GeoJSON files for whole slide images (WSIs)."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from PIL.ImageDraw import _Ink
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.annotations import PolygonMask, XMLPolygonMask
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


class AnnotationMask(XMLPolygonMask):
    def __init__(
        self,
        path: str | Path,
        mask_size: tuple[int, int],
        mask_mpp_x: float,
        mask_mpp_y: float,
        annotation_mpp: tuple[float, float],
        group_names: str | list[str] | None = None,
        mode: str = "P",
    ):
        super().__init__(path, mask_size, mask_mpp_x, mask_mpp_y, mode)
        self._annotation_mpp_x, self._annotation_mpp_y = annotation_mpp
        self.group_names = [group_names] if isinstance(group_names, str) else group_names

    @property
    def regions(self) -> Iterable[tuple[ET.Element, _Ink]]:
        for region in self.root.findall(".//Annotation"):
            if self.group_names is None or region.get("PartOfGroup") in self.group_names:
                yield region, 255

    def get_region_coordinates(
        self, region: ET.Element
    ) -> Iterable[tuple[float, float]]:
        for vertex in region.findall("Coordinates/Coordinate"):
            x_coord = vertex.get("X")
            y_coord = vertex.get("Y")
            if x_coord is None or y_coord is None:
                raise ValueError(
                    f"Invalid coordinate in XML: {vertex}. Expected 'X' and 'Y' attributes."
                )

            yield float(x_coord), float(y_coord)

    @property
    def annotation_mpp_x(self) -> float:
        return self._annotation_mpp_x

    @property
    def annotation_mpp_y(self) -> float:
        return self._annotation_mpp_y


class GeoJSONAnnotationMask(PolygonMask[list[tuple[float, float]]]):
    def __init__(
        self,
        path: str | Path,
        mask_size: tuple[int, int],
        mask_mpp_x: float,
        mask_mpp_y: float,
        annotation_mpp: tuple[float, float],
        group_names: str | list[str] | None = None,
        mode: str = "P",
    ):
        super().__init__(mask_size, mask_mpp_x, mask_mpp_y, mode)
        self._annotation_mpp_x, self._annotation_mpp_y = annotation_mpp
        self.group_names = [group_names] if isinstance(group_names, str) else group_names

        with open(path) as f:
            data = json.load(f)

        # QuPath's GeoJSON export can be either a bare list of Feature objects
        # or a proper FeatureCollection ({"type": "FeatureCollection", "features": [...]})
        # depending on the export options used, so accept both.
        self.features: list[dict[str, Any]] = (
            data if isinstance(data, list) else data["features"]
        )

    @property
    def regions(self) -> Iterable[tuple[list[tuple[float, float]], _Ink]]:
        for feature in self.features:
            classification = feature.get("properties", {}).get("classification", {})
            if self.group_names is not None and classification.get("name") not in self.group_names:
                continue

            geometry = feature["geometry"]
            if geometry["type"] == "Polygon":
                polygons = [geometry["coordinates"]]
            elif geometry["type"] == "MultiPolygon":
                polygons = geometry["coordinates"]
            else:
                # Skip non-polygonal geometries (Point, LineString, ...)
                continue

            for polygon in polygons:
                exterior_ring = polygon[0]  # holes (subsequent rings) are ignored
                yield [(x, y) for x, y in exterior_ring], 255

    def get_region_coordinates(
        self, region: list[tuple[float, float]]
    ) -> Iterable[tuple[float, float]]:
        return region

    @property
    def annotation_mpp_x(self) -> float:
        return self._annotation_mpp_x

    @property
    def annotation_mpp_y(self) -> float:
        return self._annotation_mpp_y


ANNOTATION_MASK_TYPES: dict[str, type[AnnotationMask | GeoJSONAnnotationMask]] = {
    ".xml": AnnotationMask,
    ".geojson": GeoJSONAnnotationMask,
}


@ray.remote
def process_slide(
    slide_path: Path,
    level: int,
    output_path: Path,
    annotation_map: dict[str, Path],
    annotation_groups: dict[str, str | list[str]],
) -> None:

    # no annotation file for given slide (assuming slide and corresponding annotation file share the stem)
    if slide_path.stem not in annotation_map:
        return

    annotation_file = annotation_map[slide_path.stem]

    mask_cls = ANNOTATION_MASK_TYPES.get(annotation_file.suffix.lower())
    if mask_cls is None:
        raise ValueError(f"Unsupported annotation file format: {annotation_file.suffix}")

    with OpenSlide(slide_path) as slide:
        tissue_mpp_x, tissue_mpp_y = slide_resolution(slide, level=level)
        annotation_mpp = slide_resolution(slide, level=0)
        mask_size = slide.level_dimensions[level]

    for mask_type, group_names in annotation_groups.items():
        annotator = mask_cls(
            path=annotation_file,
            mask_size=mask_size,
            mask_mpp_x=tissue_mpp_x,
            mask_mpp_y=tissue_mpp_y,
            annotation_mpp=annotation_mpp,
            group_names=group_names,
        )

        # Get the mask from the annotator
        mask = cast("pyvips.Image", pyvips.Image.new_from_array(annotator()))
        # If the mask is empty (all pixels are 0), skip saving
        if mask.max() == 0:
            continue

        # Save the mask to the destination directory
        mask_path = output_path / mask_type / slide_path.with_suffix(".tiff").name
        mask_path.parent.mkdir(exist_ok=True, parents=True)
        write_big_tiff(
            image=mask,
            path=mask_path,
            mpp_x=tissue_mpp_x,
            mpp_y=tissue_mpp_y,
        )


@with_cli_args(["+preprocessing=annot_masks"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger is not None, "Need logger"

    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(mlflow.artifacts.download_artifacts(config.data.metadata_table))
    annotation_files = [
        f
        for pattern in ("*.xml", "*.geojson")
        for f in Path(config.annotation_dir).rglob(pattern)
    ]
    annotation_map = {f.stem: f for f in annotation_files}

    slides_path = [Path(path) for path in df["slide_path"]]

    process_items(
        slides_path,
        process_item=process_slide,
        fn_kwargs={
            "level": config.level,
            "output_path": output_path,
            "annotation_map": annotation_map,
            "annotation_groups": config.annotation_groups,
        },
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(local_dir=str(output_path), artifact_path="annotation_masks")


if __name__ == "__main__":
    main()
