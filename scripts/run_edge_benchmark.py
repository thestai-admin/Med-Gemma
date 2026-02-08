#!/usr/bin/env python3
"""
Run edge model benchmarks comparing GPU vs CPU inference.

Usage:
    python scripts/run_edge_benchmark.py [--model-path models/edge/medsiglip_int8.onnx]
                                          [--n-samples 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_dataset(n_samples: int = 20):
    """Generate synthetic test images and labels for benchmarking."""
    images = []
    labels = []
    for i in range(n_samples):
        # Create synthetic CXR-sized images
        img = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        images.append(img)
        labels.append(i % 2)  # alternating normal/pneumonia
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Edge model benchmark")
    parser.add_argument(
        "--model-path",
        default="models/edge/medsiglip_int8.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of synthetic samples",
    )
    args = parser.parse_args()

    from src.edge.inference import EdgeClassifier
    from src.edge.benchmark import run_edge_benchmark

    # Load model
    print(f"Loading edge model: {args.model_path}")
    classifier = EdgeClassifier(args.model_path)
    print(f"Model size: {classifier.model_size_mb:.1f} MB")

    # Generate test data
    print(f"Generating {args.n_samples} synthetic samples...")
    images, labels = generate_synthetic_dataset(args.n_samples)

    # Run benchmark
    print("Running benchmark...")
    result = run_edge_benchmark(
        classifier, images, labels,
        model_type="edge_int8",
    )

    # Print results
    print()
    print("=" * 50)
    print("EDGE BENCHMARK RESULTS")
    print("=" * 50)
    for key, value in result.to_dict().items():
        if key != "latencies":
            print(f"  {key:<25} {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
