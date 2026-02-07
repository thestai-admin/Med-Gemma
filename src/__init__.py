# MedGemma Impact Challenge - Source Package
"""
Core source code for the MedGemma Impact Challenge competition entry.

Modules:
- model: HAI-DEF model wrappers (MedGemma, MedSigLIP)
- data: Data loading utilities for medical datasets
- inference: High-level inference pipeline
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
    # Data
    "NIHChestXray",
    "ChestXrayPneumonia",
    "MedQA",
    "DatasetIDs",
    "get_sample_chest_xrays",
    "get_sample_medical_questions",
    # Inference
    "MedicalImageAnalyzer",
    "AnalysisType",
    "AnalysisResult",
    "ChestXrayReport",
    "analyze_chest_xray",
    "generate_xray_report",
    "ask_medical_question",
]


_LAZY_EXPORTS = {
    # Models
    "MedGemma": ("src.model", "MedGemma"),
    "MedSigLIP": ("src.model", "MedSigLIP"),
    "ModelIDs": ("src.model", "ModelIDs"),
    "PROMPTS": ("src.model", "PROMPTS"),
    "get_medgemma": ("src.model", "get_medgemma"),
    "get_medsiglip": ("src.model", "get_medsiglip"),
    # Data
    "NIHChestXray": ("src.data", "NIHChestXray"),
    "ChestXrayPneumonia": ("src.data", "ChestXrayPneumonia"),
    "MedQA": ("src.data", "MedQA"),
    "DatasetIDs": ("src.data", "DatasetIDs"),
    "get_sample_chest_xrays": ("src.data", "get_sample_chest_xrays"),
    "get_sample_medical_questions": ("src.data", "get_sample_medical_questions"),
    # Inference
    "MedicalImageAnalyzer": ("src.inference", "MedicalImageAnalyzer"),
    "AnalysisType": ("src.inference", "AnalysisType"),
    "AnalysisResult": ("src.inference", "AnalysisResult"),
    "ChestXrayReport": ("src.inference", "ChestXrayReport"),
    "analyze_chest_xray": ("src.inference", "analyze_chest_xray"),
    "generate_xray_report": ("src.inference", "generate_xray_report"),
    "ask_medical_question": ("src.inference", "ask_medical_question"),
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
