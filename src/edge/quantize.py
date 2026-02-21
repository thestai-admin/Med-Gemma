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

    text_inputs = processor(text=labels, return_tensors="pt", padding="max_length")
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


class _VisionWithProjection:
    """Wrapper that runs vision_model + L2 norm (SigLIP has no projection layer)."""
    pass


def export_medsiglip_onnx(
    output_path: str,
    model_id: str = "google/medsiglip-448",
    save_text_embeddings: bool = True,
) -> str:
    """
    Export MedSigLIP vision encoder + projection to ONNX format.

    Exports the vision encoder with L2 normalization, so the ONNX
    output is already in the shared embedding space (matching
    get_image_features output). SigLIP has no separate projection layer.

    Args:
        output_path: Path for the output ONNX file
        model_id: HuggingFace model ID
        save_text_embeddings: Whether to save pre-computed text embeddings

    Returns:
        Path to the exported ONNX file
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoProcessor

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MedSigLIP from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    # Wrapper that mirrors model.get_image_features():
    #   vision_model -> pooler_output -> L2 normalize
    # IMPORTANT: Must use return_dict=False + positional indexing [1]
    # so ONNX tracing captures the correct output tensor.
    # With return_dict=True (default), the BaseModelOutputWithPooling
    # dict-like object doesn't trace correctly to ONNX.
    class VisionWrapper(nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model

        def forward(self, pixel_values):
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                return_dict=False,
            )
            # Index [1] = pooler_output (attention-pooled features)
            pooled_output = vision_outputs[1]
            image_features = pooled_output / pooled_output.norm(
                dim=-1, keepdim=True
            )
            return image_features

    wrapper = VisionWrapper(model.vision_model)
    wrapper.eval()

    # Create dummy input matching processor output
    from PIL import Image
    dummy_image = Image.new("RGB", (448, 448))
    pixel_values = processor(images=dummy_image, return_tensors="pt")["pixel_values"]

    # Verify wrapper matches get_image_features (PyTorch)
    with torch.no_grad():
        wrapper_out = wrapper(pixel_values)
        direct_out = model.get_image_features(pixel_values)
        direct_out = direct_out / direct_out.norm(dim=-1, keepdim=True)
        match = torch.allclose(wrapper_out, direct_out, atol=1e-5)
        print(f"Wrapper matches get_image_features: {match}")

    print(f"Exporting vision encoder to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        (pixel_values,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Verify ONNX output matches PyTorch output
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_out = ort_session.run(None, {"pixel_values": pixel_values.numpy()})[0]
    onnx_match = np.allclose(wrapper_out.numpy(), onnx_out, atol=1e-4)
    max_diff = np.max(np.abs(wrapper_out.numpy() - onnx_out))
    print(f"ONNX matches PyTorch: {onnx_match} (max diff: {max_diff:.6f})")
    if not onnx_match:
        print(f"WARNING: ONNX output diverges from PyTorch (max diff={max_diff:.4f})")

    # Extract SigLIP learned logit_scale and logit_bias for edge inference.
    # The GPU path uses model(**inputs) which internally applies these;
    # the edge path must apply them explicitly after cosine similarity.
    logit_scale = float(model.logit_scale.exp().item())
    logit_bias = float(model.logit_bias.item())
    print(f"SigLIP logit_scale={logit_scale:.4f}, logit_bias={logit_bias:.4f}")

    # Save preprocessor config for edge inference
    import json
    preproc_config = {
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": 448,
        "rescale_factor": 1.0 / 255.0,
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
    }
    config_path = output_path.parent / "preprocess_config.json"
    with open(str(config_path), "w") as f:
        json.dump(preproc_config, f, indent=2)
    print(f"Saved preprocessor config: {config_path}")

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
    import onnxruntime as ort
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

    # Verify INT8 output matches FP32 (cosine similarity)
    fp32_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    test_input = np.random.RandomState(42).randn(1, 3, 448, 448).astype(np.float32)
    fp32_out = fp32_sess.run(None, {"pixel_values": test_input})[0].flatten()
    int8_out = int8_sess.run(None, {"pixel_values": test_input})[0].flatten()
    cosim = float(np.dot(fp32_out, int8_out) / (
        np.linalg.norm(fp32_out) * np.linalg.norm(int8_out) + 1e-8))
    print(f"INT8 vs FP32 cosine similarity: {cosim:.6f}")
    if cosim < 0.99:
        print(f"WARNING: INT8 diverges from FP32 (cosim={cosim:.4f}). "
              "Attention pooling is sensitive to INT8 quantization. "
              "Use quantize_onnx_selective_int8() to exclude attention nodes, "
              "or use FP32 ONNX for accurate edge inference.")

    return str(output_path)


def quantize_onnx_selective_int8(
    onnx_path: str,
    output_path: str,
    exclude_keywords: Optional[List[str]] = None,
) -> str:
    """
    Quantize an ONNX model to INT8, excluding attention and normalization nodes.

    Standard dynamic INT8 degrades SigLIP-style attention pooling because
    pooling attention weights are precision-sensitive. This function inspects
    the model graph and excludes nodes whose names contain attention- or
    normalization-related keywords before quantizing, preserving accuracy
    while still reducing model size for linear projection layers.

    Args:
        onnx_path: Path to the FP32 ONNX model
        output_path: Path for the quantized output model
        exclude_keywords: Node name substrings to exclude from quantization.
            Defaults to ["attn", "attention", "pool", "norm", "layer_norm"].

    Returns:
        Path to the quantized model
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if exclude_keywords is None:
        exclude_keywords = ["attn", "attention", "pool", "norm", "layer_norm"]

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Inspect model graph to find precision-sensitive nodes to skip
    model = onnx.load(str(onnx_path))
    nodes_to_exclude = [
        node.name
        for node in model.graph.node
        if any(kw in node.name.lower() for kw in exclude_keywords)
    ]
    print(f"Selective INT8: excluding {len(nodes_to_exclude)} attention/norm nodes "
          f"out of {len(model.graph.node)} total")

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
    )

    # Report size reduction and verify accuracy
    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quantized_size = output_path_obj.stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    print(f"Original: {original_size:.1f} MB  →  Selective INT8: {quantized_size:.1f} MB  ({reduction:.1f}% reduction)")

    fp32_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    test_input = np.random.RandomState(42).randn(1, 3, 448, 448).astype(np.float32)
    fp32_out = fp32_sess.run(None, {"pixel_values": test_input})[0].flatten()
    int8_out = int8_sess.run(None, {"pixel_values": test_input})[0].flatten()
    cosim = float(np.dot(fp32_out, int8_out) / (
        np.linalg.norm(fp32_out) * np.linalg.norm(int8_out) + 1e-8))
    print(f"Selective INT8 vs FP32 cosine similarity: {cosim:.6f}")
    if cosim < 0.99:
        print(f"WARNING: Selective INT8 still diverges (cosim={cosim:.4f}). "
              "Consider expanding exclude_keywords or using FP32 ONNX.")

    return str(output_path)
