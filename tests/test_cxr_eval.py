"""Tests for deterministic CXR evaluation helpers."""

from types import SimpleNamespace

from src.eval import (
    bootstrap_metric_ci,
    compute_auroc,
    compute_binary_metrics,
    confusion_counts,
    evaluate_scores,
    profile_orchestrator_latency,
    select_threshold,
    sweep_thresholds,
)


def test_confusion_counts_and_binary_metrics():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]

    counts = confusion_counts(y_true, y_pred)
    assert (counts.tp, counts.tn, counts.fp, counts.fn) == (1, 1, 1, 1)

    metrics = compute_binary_metrics(y_true, y_pred, threshold=0.5)
    assert metrics.accuracy == 0.5
    assert metrics.f1 == 0.5


def test_threshold_sweep_and_recall_priority_selection():
    y_true = [0, 0, 1, 1, 1]
    scores = [0.1, 0.3, 0.4, 0.7, 0.9]

    results = sweep_thresholds(y_true, scores, thresholds=[0.2, 0.5, 0.8])
    selected = select_threshold(results, objective="recall_priority")

    assert selected.threshold == 0.2
    assert selected.metrics.recall == 1.0


def test_bootstrap_metric_ci_returns_probability_bounds():
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 0, 1, 0, 1, 1]

    lower, upper = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric="f1",
        n_bootstrap=200,
        seed=7,
    )

    assert 0.0 <= lower <= upper <= 1.0


def test_evaluate_scores_applies_threshold_correctly():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.55, 0.9]

    metrics = evaluate_scores(y_true, scores, threshold=0.6)
    assert metrics.counts.tp == 1
    assert metrics.counts.fp == 1


def test_compute_auroc_perfect_classifier():
    """Perfect classifier (scores match labels) should give AUROC=1.0."""
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    auroc = compute_auroc(y_true, scores)
    assert abs(auroc - 1.0) < 1e-6


def test_compute_auroc_random_classifier():
    """Random classifier (scores independent of labels) should give AUROC≈0.5."""
    import numpy as np
    rng = np.random.default_rng(0)
    y_true = [int(x) for x in rng.integers(0, 2, 200)]
    scores = [float(x) for x in rng.random(200)]
    auroc = compute_auroc(y_true, scores)
    assert 0.3 < auroc < 0.7  # loose bound for small sample randomness


def test_compute_auroc_range():
    """AUROC must lie in [0, 1]."""
    y_true = [0, 1, 0, 1, 1]
    scores = [0.3, 0.7, 0.6, 0.2, 0.9]
    auroc = compute_auroc(y_true, scores)
    assert 0.0 <= auroc <= 1.0


def test_compute_auroc_requires_both_classes():
    """AUROC with only one class should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="positive and one negative"):
        compute_auroc([1, 1, 1], [0.5, 0.6, 0.7])


def test_profile_orchestrator_latency_aggregates_runs():
    class DummyOrchestrator:
        def run(self, **kwargs):
            return SimpleNamespace(timings={"imaging": 0.2, "reasoning": 0.3, "total": 0.7})

    result = profile_orchestrator_latency(
        DummyOrchestrator(),
        run_kwargs={"chief_complaint": "cough"},
        repeats=3,
    )

    assert result.runs == 3
    assert result.median_by_stage["imaging"] == 0.2
    assert result.p95_by_stage["total"] == 0.7
