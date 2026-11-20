from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    Metric,
    MetricCollection,
    NegativePredictiveValue,
    Precision,
    Recall,
    Specificity,
)


def thresholded_metrics(threshold: float = 0.5) -> dict[str, Metric]:
    """Binary metrics that depend on the decision threshold."""
    return {
        "accuracy": Accuracy("binary", threshold=threshold),
        "precision": Precision("binary", threshold=threshold),
        "recall": Recall("binary", threshold=threshold),
        "specificity": Specificity("binary", threshold=threshold),
        "negative_predictive_value": NegativePredictiveValue(
            "binary", threshold=threshold
        ),
    }


def binary_metrics(threshold: float | None) -> dict[str, Metric | MetricCollection]:
    """AUC/AUPRC plus the thresholded metrics.

    With `threshold=None` (the model's thresholds aren't chosen yet) only the
    threshold-free AUC/AUPRC are returned. The rest can be added to the run
    later with `postprocessing.threshold_metrics`.
    """
    metrics: dict[str, Metric | MetricCollection] = {
        "AUC": AUROC("binary"),
        "AUPRC": AveragePrecision("binary"),
    }
    if threshold is not None:
        metrics |= thresholded_metrics(threshold)
    return metrics
