# MedGemma Impact Challenge - Source Package
"""
Core source code for the MedGemma Impact Challenge competition entry.

Modules:
- model: HAI-DEF model wrappers (MedGemma, MedSigLIP)
- agents: Multi-agent diagnostic pipeline
- eval: Reproducible CXR evaluation utilities
"""

__version__ = "0.1.0"

from importlib import import_module

__all__ = [
    # Models
    "MedGemma",
    "MedSigLIP",
    "ModelIDs",
    "PROMPTS",
    "get_medgemma",
    "get_medsiglip",
    # Evaluation
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
    # Edge AI
    "EdgeClassifier",
    "EdgeBenchmarkResult",
]


_LAZY_EXPORTS = {
    # Models
    "MedGemma": ("src.model", "MedGemma"),
    "MedSigLIP": ("src.model", "MedSigLIP"),
    "ModelIDs": ("src.model", "ModelIDs"),
    "PROMPTS": ("src.model", "PROMPTS"),
    "get_medgemma": ("src.model", "get_medgemma"),
    "get_medsiglip": ("src.model", "get_medsiglip"),
    # Evaluation
    "ConfusionCounts": ("src.eval", "ConfusionCounts"),
    "EvalMetrics": ("src.eval", "EvalMetrics"),
    "ThresholdResult": ("src.eval", "ThresholdResult"),
    "LatencyMetrics": ("src.eval", "LatencyMetrics"),
    "confusion_counts": ("src.eval", "confusion_counts"),
    "compute_binary_metrics": ("src.eval", "compute_binary_metrics"),
    "evaluate_scores": ("src.eval", "evaluate_scores"),
    "sweep_thresholds": ("src.eval", "sweep_thresholds"),
    "select_threshold": ("src.eval", "select_threshold"),
    "bootstrap_metric_ci": ("src.eval", "bootstrap_metric_ci"),
    "profile_orchestrator_latency": ("src.eval", "profile_orchestrator_latency"),
    # Edge AI
    "EdgeClassifier": ("src.edge", "EdgeClassifier"),
    "EdgeBenchmarkResult": ("src.edge", "EdgeBenchmarkResult"),
}


def __getattr__(name):
    """Lazily load heavy submodules so lightweight imports don't require ML deps."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
