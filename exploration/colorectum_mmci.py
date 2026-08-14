"""C20-24 Data Exploration."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


def stem_to_case_id(stem: str) -> str:
    # stem is of form YYYY_NNNN-XX-C+
    return stem.split("-")[0][2:].replace("_", "/")  # YY/NNNN


def stem_to_carcinoma(stem: str) -> bool:
    # stem is of form YYYY_NNNN-XX-C+
    return stem.split("-")[-1].startswith(
        "T"
    )  # possible positive tags are T, T1, T2, ...


@hydra.main(
    config_path="../configs",
    config_name="exploration/colorectum/mmci",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    for slides_dir, data_name in [
        (config.c20_24_dir, "c20_24"),
        (config.adopt_dir, "adopt"),
    ]:
        all_slides = list(Path(slides_dir).glob("*mrxs"))

        if data_name == "c20_24":
            all_slides = filter(lambda x: x.stem not in config.c20_24_blacklist, all_slides)

        all_slides = [
            slide
            for slide in all_slides
            if not slide.stem.split("-")[-1].startswith("M")
        ]  # remove metastases
        cases = [stem_to_case_id(slide.stem) for slide in all_slides]
        carcinoma_labels = [stem_to_carcinoma(slide.stem) for slide in all_slides]
        data = {
            "slide_path": all_slides,
            "case_id": cases,
            "carcinoma": carcinoma_labels,
        }

        df = pd.DataFrame(data)

        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / f"{data_name}_metadata.csv"
            df.to_csv(str(target), index=False)
            mlflow.log_artifact(str(target))


if __name__ == "__main__":
    main()
