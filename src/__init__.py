# MedGemma Impact Challenge - Source Package
"""
Core source code for the MedGemma Impact Challenge competition entry.

Modules:
- model: HAI-DEF model wrappers (MedGemma, MedSigLIP)
- data: Data loading utilities for medical datasets
- inference: High-level inference pipeline
"""

__version__ = "0.1.0"

from .model import MedGemma, MedSigLIP, ModelIDs, PROMPTS, get_medgemma, get_medsiglip
from .data import (
    NIHChestXray,
    ChestXrayPneumonia,
    MedQA,
    DatasetIDs,
    get_sample_chest_xrays,
    get_sample_medical_questions,
)
from .inference import (
    MedicalImageAnalyzer,
    AnalysisType,
    AnalysisResult,
    ChestXrayReport,
    analyze_chest_xray,
    generate_xray_report,
    ask_medical_question,
)

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
