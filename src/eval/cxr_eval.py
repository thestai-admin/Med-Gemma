"""
Deterministic evaluation utilities for CXR binary classification workflows.

This module keeps metric calculations and threshold selection out of notebooks so
reported numbers can be reproduced from code.
"""

from dataclasses import dataclass
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConfusionCounts:
    """Binary confusion matrix counts."""

    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class EvalMetrics:
    """Binary classification metrics for a fixed threshold."""

    threshold: float
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    counts: ConfusionCounts

    def to_dict(self) -> Dict[str, float]:
        """Serialize metrics as a flat dictionary."""
        return {
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "tp": self.counts.tp,
            "tn": self.counts.tn,
            "fp": self.counts.fp,
            "fn": self.counts.fn,
        }


@dataclass
class ThresholdResult:
    """Metrics result tied to a specific score threshold."""

    threshold: float
    metrics: EvalMetrics


@dataclass
class LatencyMetrics:
    """Aggregate latency profile across repeated pipeline runs."""

    runs: int
    raw_timings: Dict[str, List[float]]
    median_by_stage: Dict[str, float]
    p95_by_stage: Dict[str, float]


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division helper used by all metric calculations."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _as_array(values: Sequence[int]) -> np.ndarray:
    """Convert labels to an integer numpy array."""
    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array of labels.")
    return arr


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionCounts:
    """Compute confusion matrix counts for binary labels {0,1}."""
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    return ConfusionCounts(tp=tp, tn=tn, fp=fp, fn=fn)


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    threshold: float = 0.5,
) -> EvalMetrics:
    """Compute binary metrics from true and predicted binary labels."""
    counts = confusion_counts(y_true, y_pred)
    n_samples = int(len(y_true))

    accuracy = _safe_div(counts.tp + counts.tn, n_samples)
    precision = _safe_div(counts.tp, counts.tp + counts.fp)
    recall = _safe_div(counts.tp, counts.tp + counts.fn)
    specificity = _safe_div(counts.tn, counts.tn + counts.fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return EvalMetrics(
        threshold=threshold,
        n_samples=n_samples,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        counts=counts,
    )


def evaluate_scores(
    y_true: Sequence[int],
    pneumonia_scores: Sequence[float],
    threshold: float = 0.5,
) -> EvalMetrics:
    """Convert probability scores to labels and compute metrics."""
    y_true_arr = _as_array(y_true)
    scores = np.asarray(pneumonia_scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("Expected a 1D array of prediction scores.")
    if y_true_arr.shape != scores.shape:
        raise ValueError("y_true and pneumonia_scores must have the same shape.")

    y_pred = (scores >= threshold).astype(np.int32)
    return compute_binary_metrics(y_true_arr, y_pred, threshold=threshold)


def sweep_thresholds(
    y_true: Sequence[int],
    pneumonia_scores: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> List[ThresholdResult]:
    """Evaluate metrics across a list of thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    results: List[ThresholdResult] = []
    for threshold in thresholds:
        metrics = evaluate_scores(y_true, pneumonia_scores, threshold=float(threshold))
        results.append(ThresholdResult(threshold=float(threshold), metrics=metrics))
    return results


def select_threshold(
    threshold_results: Sequence[ThresholdResult],
    objective: str = "balanced",
) -> ThresholdResult:
    """
    Pick the best threshold for a specific operating objective.

    Supported objectives:
    - balanced: maximize F1, then recall, then specificity
    - recall_priority: maximize recall, then F1, then specificity
    """
    if not threshold_results:
        raise ValueError("threshold_results is empty.")

    objective_normalized = objective.strip().lower()
    if objective_normalized == "balanced":
        key_fn: Callable[[ThresholdResult], Tuple[float, float, float]] = (
            lambda item: (item.metrics.f1, item.metrics.recall, item.metrics.specificity)
        )
    elif objective_normalized == "recall_priority":
        key_fn = lambda item: (item.metrics.recall, item.metrics.f1, item.metrics.specificity)
    else:
        raise ValueError("objective must be 'balanced' or 'recall_priority'.")

    return max(threshold_results, key=key_fn)


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: str = "f1",
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.95,
) -> Tuple[float, float]:
    """Estimate bootstrap confidence interval for a binary metric."""
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0.")

    metric_name = metric.strip().lower()
    supported_metrics = {"accuracy", "precision", "recall", "specificity", "f1"}
    if metric_name not in supported_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Use one of: {sorted(supported_metrics)}")

    rng = np.random.default_rng(seed)
    n = len(y_true_arr)
    values: List[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        sampled_true = y_true_arr[indices]
        sampled_pred = y_pred_arr[indices]
        sampled_metrics = compute_binary_metrics(sampled_true, sampled_pred)
        values.append(getattr(sampled_metrics, metric_name))

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return float(np.quantile(values, lower_q)), float(np.quantile(values, upper_q))


def compute_auroc(
    y_true: Sequence[int],
    scores: Sequence[float],
) -> float:
    """
    Compute Area Under the ROC Curve (AUROC) using the trapezoidal rule.

    Pure numpy implementation — no sklearn dependency.

    Args:
        y_true: Ground-truth binary labels (0 or 1)
        scores: Predicted probability scores for the positive class

    Returns:
        AUROC value in [0, 1]
    """
    y_true_arr = _as_array(y_true)
    scores_arr = np.asarray(scores, dtype=np.float32)

    if y_true_arr.shape != scores_arr.shape:
        raise ValueError("y_true and scores must have the same shape.")

    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires at least one positive and one negative sample.")

    # Sort by score descending to build the ROC curve
    order = np.argsort(-scores_arr)
    sorted_labels = y_true_arr[order]

    tpr_points: List[float] = [0.0]
    fpr_points: List[float] = [0.0]
    tp, fp = 0, 0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)
    tpr_points.append(1.0)
    fpr_points.append(1.0)

    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(tpr_points, fpr_points))


def profile_orchestrator_latency(
    orchestrator,
    run_kwargs: Dict[str, object],
    repeats: int = 3,
) -> LatencyMetrics:
    """
    Profile latency from PrimaCareOrchestrator.run profile outputs.

    Notes:
    - `run_kwargs` should include minimal deterministic inputs.
    - `profile=True` is injected automatically.
    """
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    raw: Dict[str, List[float]] = {}
    for _ in range(repeats):
        result = orchestrator.run(profile=True, **run_kwargs)
        for stage, elapsed in result.timings.items():
            raw.setdefault(stage, []).append(float(elapsed))

    median_by_stage = {stage: float(median(vals)) for stage, vals in raw.items() if vals}
    p95_by_stage = {
        stage: float(np.quantile(vals, 0.95))
        for stage, vals in raw.items()
        if vals
    }

    return LatencyMetrics(
        runs=repeats,
        raw_timings=raw,
        median_by_stage=median_by_stage,
        p95_by_stage=p95_by_stage,
    )
