"""
ONNX Export and INT8 Quantization for MedSigLIP

Exports the MedSigLIP vision encoder to ONNX format and quantizes
it to INT8 for CPU-only inference in resource-limited settings.

Usage:
    from src.edge.quantize import export_medsiglip_onnx, quantize_onnx_int8

    export_medsiglip_onnx("models/edge/medsiglip_fp32.onnx")
    quantize_onnx_int8(
        "models/edge/medsiglip_fp32.onnx",
        "models/edge/medsiglip_int8.onnx",
    )
"""

from pathlib import Path
from typing import Optional, List

import numpy as np


def _compute_text_embeddings(
    model_id: str = "google/medsiglip-448",
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Pre-compute text embeddings for binary pneumonia labels.

    These are saved alongside the ONNX model so the edge classifier
    only needs to run the vision encoder at inference time.

    Args:
        model_id: HuggingFace model ID for MedSigLIP
        labels: Text labels to embed (defaults to binary pneumonia labels)

    Returns:
        numpy array of shape (n_labels, embed_dim)
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    if labels is None:
        labels = [
            "normal healthy chest x-ray with clear lungs and no infiltrates",
            "chest x-ray showing pneumonia, infection, consolidation, or infiltrates",
        ]

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    text_inputs = processor(text=labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def export_medsiglip_onnx(
    output_path: str,
    model_id: str = "google/medsiglip-448",
    save_text_embeddings: bool = True,
) -> str:
    """
    Export MedSigLIP vision encoder to ONNX format.

    Args:
        output_path: Path for the output ONNX file
        model_id: HuggingFace model ID
        save_text_embeddings: Whether to save pre-computed text embeddings

    Returns:
        Path to the exported ONNX file
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MedSigLIP from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    # Get vision encoder
    vision_model = model.vision_model

    # Create dummy input matching processor output
    from PIL import Image
    dummy_image = Image.new("RGB", (448, 448))
    pixel_values = processor(images=dummy_image, return_tensors="pt")["pixel_values"]

    print(f"Exporting vision encoder to ONNX: {output_path}")
    torch.onnx.export(
        vision_model,
        (pixel_values,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=14,
    )

    # Save text embeddings alongside ONNX model
    if save_text_embeddings:
        embeddings_path = output_path.parent / "text_embeddings.npy"
        print(f"Pre-computing text embeddings: {embeddings_path}")
        text_embeddings = _compute_text_embeddings(model_id)
        np.save(str(embeddings_path), text_embeddings)

    print(f"Export complete: {output_path}")
    return str(output_path)


def quantize_onnx_int8(
    onnx_path: str,
    output_path: str,
) -> str:
    """
    Quantize an ONNX model to INT8 using dynamic quantization.

    Args:
        onnx_path: Path to the FP32 ONNX model
        output_path: Path for the quantized INT8 model

    Returns:
        Path to the quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing {onnx_path} -> {output_path} (INT8 dynamic)")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    # Report size reduction
    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"Original: {original_size:.1f} MB")
    print(f"Quantized: {quantized_size:.1f} MB")
    print(f"Reduction: {reduction:.1f}%")

    return str(output_path)
