"""Compute slide/case/tile counts and percentage-column histograms for a data source.

Slide/case counts are derived once from `data.metadata_table`, since the
slide set is shared across the 512/224/full/filtered tiling variants --
deriving it per tiling dataset would just duplicate the same numbers. Tiling
datasets are only used for tile-level counts and percentage histograms.
"""

import tempfile
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def resolve_roi_threshold(
    thresholds: DictConfig | None, tiles: pd.DataFrame
) -> tuple[str, float] | None:
    """Pick the percentage column and threshold that define tile-level positivity.

    Prefers `carcinoma_roi_t` / `carcinoma_roi_percentage`, falling back to
    `epithelium_roi_t` / `epithelium_roi_percentage` when the carcinoma ROI
    threshold or column is not available for this data source.

    Arguments:
        thresholds (DictConfig | None): `data.thresholds` config, if present.
        tiles (pd.DataFrame): tiles.parquet contents.

    Returns:
        tuple[str, float] | None: (percentage column, threshold) to use, or
            None if neither source is fully available.
    """
    if thresholds is None:
        return None

    for threshold_key, column in (
        ("carcinoma_roi_t", "carcinoma_roi_percentage"),
        ("epithelium_roi_t", "epithelium_roi_percentage"),
    ):
        threshold = thresholds.get(threshold_key)
        if threshold is not None and column in tiles.columns:
            return column, threshold

    return None


def slide_stats(metadata: pd.DataFrame) -> dict[str, int]:
    """Count slides overall and by the boolean `carcinoma` flag.

    Arguments:
        metadata (pd.DataFrame): `data.metadata_table` contents (one row per slide).

    Returns:
        dict[str, int]: Slide counts.
    """
    positive = metadata["carcinoma"]
    return {
        "num_slides": len(metadata),
        "num_slides_carcinoma_positive": int(positive.sum()),
        "num_slides_carcinoma_negative": int((~positive).sum()),
    }


def case_stats(metadata: pd.DataFrame) -> dict[str, int]:
    """Count cases overall and by positivity, if `case_id` is present.

    A case is positive if at least one of its slides is positive.

    Arguments:
        metadata (pd.DataFrame): `data.metadata_table` contents (one row per slide).

    Returns:
        dict[str, int]: Case counts, or an empty dict if `case_id` isn't present.
    """
    if "case_id" not in metadata.columns:
        return {}

    case_positive = metadata.groupby("case_id")["carcinoma"].any()
    return {
        "num_cases": len(case_positive),
        "num_cases_positive": int(case_positive.sum()),
        "num_cases_negative": int((~case_positive).sum()),
    }


def tile_stats(tiles: pd.DataFrame, thresholds: DictConfig | None) -> dict[str, int]:
    """Count tiles overall and, if a ROI threshold/column is available, by positivity.

    Arguments:
        tiles (pd.DataFrame): tiles.parquet contents.
        thresholds (DictConfig | None): `data.thresholds` config, if present.

    Returns:
        dict[str, int]: Tile counts.
    """
    stats = {"num_tiles": len(tiles)}

    resolved = resolve_roi_threshold(thresholds, tiles)
    if resolved is None:
        return stats

    column, threshold = resolved
    positive = tiles[column] > threshold
    stats["num_tiles_positive"] = int(positive.sum())
    stats["num_tiles_negative"] = int((~positive).sum())

    return stats


def plot_percentage_histogram(values: pd.Series, column: str, out_dir: Path) -> Path:
    """Plot a full histogram and a non-zero-only histogram of a percentage column.

    Arguments:
        values (pd.Series): Column values to plot.
        column (str): Column name, used for titles/labels and the output filename.
        out_dir (Path): Directory to save the figure into.

    Returns:
        Path: Path to the saved figure.
    """
    nonzero = values[values > 0]

    fig, (ax_full, ax_nonzero) = plt.subplots(1, 2, figsize=(12, 5))

    ax_full.hist(values, bins=50, range=(0, 1), color="steelblue", edgecolor="black")
    ax_full.set_title(f"All tiles (n={len(values)})")
    ax_full.set_xlabel(column)
    ax_full.set_ylabel("Count")

    ax_nonzero.hist(
        nonzero, bins=50, range=(0, 1), color="darkorange", edgecolor="black"
    )
    ax_nonzero.set_title(f"Non-zero tiles (n={len(nonzero)})")
    ax_nonzero.set_xlabel(column)
    ax_nonzero.set_ylabel("Count")

    fig.suptitle(column)
    fig.tight_layout()

    path = out_dir / f"{column}.png"
    fig.savefig(path)
    plt.close(fig)

    return path


def compute_and_log_tile_stats(
    tiling_uri: str, suffix: str, thresholds: DictConfig | None, logger: MLFlowLogger
) -> None:
    """Compute tile stats and percentage histograms for one tiling dataset.

    Slide/case counts are not derived here -- see `main`, which reports them
    once from `data.metadata_table` instead.

    Arguments:
        tiling_uri (str): MLflow URI of the tiling dataset directory.
        suffix (str): Namespace suffix for logged metrics/plots (e.g. "512", "filtered_224").
        thresholds (DictConfig | None): `data.thresholds` config, if present.
        logger (MLFlowLogger): Logger used to log metrics and plot artifacts.
    """
    tiling_path = Path(mlflow.artifacts.download_artifacts(tiling_uri))
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    stats = tile_stats(tiles, thresholds)
    print(f"[{suffix}] stats:", stats)
    logger.log_metrics({f"{suffix}/{name}": value for name, value in stats.items()})

    percentage_cols = [col for col in tiles.columns if col.endswith("percentage")]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        for column in percentage_cols:
            plot_path = plot_percentage_histogram(tiles[column], column, out_dir)
            logger.log_artifact(str(plot_path), artifact_path=f"plots/{suffix}")


@with_cli_args(["+preprocessing=tiling_stats"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metadata_path = mlflow.artifacts.download_artifacts(config.data.metadata_table)
    metadata = pd.read_csv(metadata_path)

    stats = {**slide_stats(metadata), **case_stats(metadata)}
    print("slide/case stats:", stats)
    logger.log_metrics(stats)

    thresholds = config.data.get("thresholds")

    for field, suffix in (
        ("tiles_uri_512", "512"),
        ("tiles_uri_224", "224"),
        ("tiles_filtered_uri_512", "filtered_512"),
        ("tiles_filtered_uri_224", "filtered_224"),
    ):
        tiling_uri = config.data.get(field)
        if tiling_uri is None:
            continue

        compute_and_log_tile_stats(tiling_uri, suffix, thresholds, logger)


if __name__ == "__main__":
    main()
