"""Evaluation helpers for reproducible CXR benchmarking."""

from .cxr_eval import (
    ConfusionCounts,
    EvalMetrics,
    ThresholdResult,
    LatencyMetrics,
    confusion_counts,
    compute_binary_metrics,
    evaluate_scores,
    sweep_thresholds,
    select_threshold,
    bootstrap_metric_ci,
    profile_orchestrator_latency,
)

__all__ = [
    "ConfusionCounts",
    "EvalMetrics",
    "ThresholdResult",
    "LatencyMetrics",
    "confusion_counts",
    "compute_binary_metrics",
    "evaluate_scores",
    "sweep_thresholds",
    "select_threshold",
    "bootstrap_metric_ci",
    "profile_orchestrator_latency",
]
