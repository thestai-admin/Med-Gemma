#!/usr/bin/env python3
"""
Export MedSigLIP to ONNX and quantize to INT8 for edge deployment.

Run on a machine with GPU access (e.g., Kaggle T4):
    python scripts/export_edge_model.py

Outputs:
    models/edge/medsiglip_fp32.onnx    - Full precision ONNX model
    models/edge/medsiglip_int8.onnx    - INT8 quantized model
    models/edge/text_embeddings.npy     - Pre-computed text embeddings
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edge.quantize import export_medsiglip_onnx, quantize_onnx_int8


def main():
    output_dir = Path("models/edge")
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = str(output_dir / "medsiglip_fp32.onnx")
    int8_path = str(output_dir / "medsiglip_int8.onnx")

    # Step 1: Export to ONNX (FP32)
    print("=" * 60)
    print("Step 1: Exporting MedSigLIP to ONNX (FP32)")
    print("=" * 60)
    export_medsiglip_onnx(fp32_path)

    # Step 2: Quantize to INT8
    print()
    print("=" * 60)
    print("Step 2: Quantizing to INT8")
    print("=" * 60)
    quantize_onnx_int8(fp32_path, int8_path)

    # Step 3: Sanity check
    print()
    print("=" * 60)
    print("Step 3: Sanity Check")
    print("=" * 60)
    try:
        from src.edge.inference import EdgeClassifier
        from PIL import Image
        import numpy as np

        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))

        classifier = EdgeClassifier(int8_path)
        result = classifier.classify_pneumonia(test_image)
        print(f"Test classification: {result}")
        print(f"Model size: {classifier.model_size_mb:.1f} MB")
        print("Sanity check passed!")
    except Exception as e:
        print(f"Sanity check failed (non-critical): {e}")

    print()
    print("Export complete! Files saved to models/edge/")


if __name__ == "__main__":
    main()
