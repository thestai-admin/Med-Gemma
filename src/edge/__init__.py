"""
Edge AI Module - CPU-Only Inference for Resource-Limited Settings

Provides ONNX-quantized MedSigLIP for chest X-ray screening on CPU,
enabling deployment in clinics without GPU infrastructure.

Tiered architecture:
- Edge (CPU): Fast binary pneumonia screening with quantized MedSigLIP
- Cloud (GPU): Full multi-agent diagnostic pipeline with MedGemma

Designed for the Edge AI Prize track in the MedGemma Impact Challenge.
"""

from .inference import EdgeClassifier
from .benchmark import EdgeBenchmarkResult, run_edge_benchmark, compare_models
from .quantize import export_medsiglip_onnx, quantize_onnx_int8, quantize_onnx_selective_int8

__all__ = [
    "EdgeClassifier",
    "EdgeBenchmarkResult",
    "run_edge_benchmark",
    "compare_models",
    "export_medsiglip_onnx",
    "quantize_onnx_int8",
    "quantize_onnx_selective_int8",
]
