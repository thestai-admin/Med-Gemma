"""
Edge AI Benchmarking Utilities

Measures latency, memory, and accuracy for edge vs GPU models
to demonstrate the feasibility of tiered deployment.

Usage:
    from src.edge.benchmark import run_edge_benchmark, compare_models

    result = run_edge_benchmark(classifier, images, labels)
    print(result.to_dict())
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EdgeBenchmarkResult:
    """Benchmark results for an edge or GPU model."""
    model_type: str  # "edge_int8", "edge_fp32", "gpu_full"
    model_size_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    memory_peak_mb: float = 0.0
    accuracy: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_samples: int = 0
    latencies: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize to dictionary."""
        return {
            "model_type": self.model_type,
            "model_size_mb": round(self.model_size_mb, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "accuracy": round(self.accuracy, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_samples": self.n_samples,
        }


def run_edge_benchmark(
    classifier,
    images: Sequence,
    labels: Sequence[int],
    n_warmup: int = 3,
    threshold: float = 0.5,
    model_type: str = "edge_int8",
) -> EdgeBenchmarkResult:
    """
    Benchmark an edge classifier on a set of images.

    Args:
        classifier: EdgeClassifier instance (or any object with classify_pneumonia())
        images: Sequence of PIL Images
        labels: Ground truth binary labels (0=normal, 1=pneumonia)
        n_warmup: Number of warmup iterations before timing
        threshold: Decision threshold for pneumonia classification
        model_type: Label for this model type in results

    Returns:
        EdgeBenchmarkResult with latency and accuracy metrics
    """
    import tracemalloc

    if len(images) == 0:
        raise ValueError("images must not be empty")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    # Warmup runs
    warmup_count = min(n_warmup, len(images))
    for i in range(warmup_count):
        classifier.classify_pneumonia(images[i])

    # Timed runs with memory tracking
    tracemalloc.start()
    latencies = []
    predictions = []

    for img in images:
        start = time.perf_counter()
        result = classifier.classify_pneumonia(img)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        predictions.append(1 if result["pneumonia"] >= threshold else 0)

    _, memory_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compute metrics
    from ..eval.cxr_eval import compute_binary_metrics
    labels_list = list(labels)
    metrics = compute_binary_metrics(labels_list, predictions, threshold=threshold)

    # Get model size
    model_size = getattr(classifier, "model_size_mb", 0.0)

    return EdgeBenchmarkResult(
        model_type=model_type,
        model_size_mb=model_size,
        avg_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        memory_peak_mb=memory_peak / (1024 * 1024),
        accuracy=metrics.accuracy,
        recall=metrics.recall,
        f1=metrics.f1,
        n_samples=len(images),
        latencies=latencies,
    )


def compare_models(
    full_result: EdgeBenchmarkResult,
    edge_result: EdgeBenchmarkResult,
) -> str:
    """
    Generate a formatted comparison table between GPU and Edge models.

    Args:
        full_result: Benchmark result from full GPU model
        edge_result: Benchmark result from edge model

    Returns:
        Formatted comparison string
    """
    lines = [
        "=" * 60,
        "MODEL COMPARISON: GPU vs Edge",
        "=" * 60,
        "",
        f"{'Metric':<25} {'GPU':>15} {'Edge (CPU)':>15}",
        "-" * 55,
        f"{'Model Size (MB)':<25} {full_result.model_size_mb:>15.1f} {edge_result.model_size_mb:>15.1f}",
        f"{'Avg Latency (ms)':<25} {full_result.avg_latency_ms:>15.1f} {edge_result.avg_latency_ms:>15.1f}",
        f"{'P95 Latency (ms)':<25} {full_result.p95_latency_ms:>15.1f} {edge_result.p95_latency_ms:>15.1f}",
        f"{'Peak Memory (MB)':<25} {full_result.memory_peak_mb:>15.1f} {edge_result.memory_peak_mb:>15.1f}",
        f"{'Accuracy':<25} {full_result.accuracy:>15.3f} {edge_result.accuracy:>15.3f}",
        f"{'Recall':<25} {full_result.recall:>15.3f} {edge_result.recall:>15.3f}",
        f"{'F1 Score':<25} {full_result.f1:>15.3f} {edge_result.f1:>15.3f}",
        f"{'Samples':<25} {full_result.n_samples:>15d} {edge_result.n_samples:>15d}",
        "",
    ]

    # Compute speedup and size reduction
    if edge_result.avg_latency_ms > 0 and full_result.avg_latency_ms > 0:
        speedup = full_result.avg_latency_ms / edge_result.avg_latency_ms
        lines.append(f"Latency ratio (GPU/Edge): {speedup:.2f}x")
    if edge_result.model_size_mb > 0 and full_result.model_size_mb > 0:
        reduction = (1 - edge_result.model_size_mb / full_result.model_size_mb) * 100
        lines.append(f"Size reduction: {reduction:.1f}%")
    if full_result.f1 > 0:
        f1_diff = edge_result.f1 - full_result.f1
        lines.append(f"F1 difference: {f1_diff:+.3f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
