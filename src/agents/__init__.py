"""
PrimaCare AI - Agentic Workflow Components

Multi-agent system for primary care diagnostic support.
Designed for the MedGemma Impact Challenge - Agentic Workflow Prize.

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Intake    │ --> │   Imaging   │ --> │  Reasoning  │ --> │ Guidelines  │
    │   Agent     │     │   Agent     │     │   Agent     │     │   Agent     │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
         │                    │                   │                    │
         v                    v                   v                    v
    Structured HPI      Image Analysis      Differential Dx     Evidence-Based
                                            + Workup            Recommendations

Agents:
- IntakeAgent: Structures patient history into HPI format
- ImagingAgent: Analyzes chest X-rays with MedGemma
- ReasoningAgent: Combines clinical + imaging for diagnosis
- GuidelinesAgent: RAG for clinical practice guidelines
- PrimaCareOrchestrator: Coordinates all agents

Usage:
    from src.agents import PrimaCareOrchestrator

    orchestrator = PrimaCareOrchestrator()
    result = orchestrator.run(
        chief_complaint="Cough for 2 weeks",
        history="65yo male, smoker, fever",
        xray_image=image
    )
"""

from .intake import IntakeAgent
from .imaging import ImagingAgent, LongitudinalAnalysis, ChangeStatus
from .reasoning import ReasoningAgent
from .guidelines import GuidelinesAgent
from .orchestrator import PrimaCareOrchestrator
from .volumetric import VolumetricImagingAgent, VolumetricAnalysis, VolumetricModality
from .ehr_navigator import EHRNavigatorAgent, EHRQueryResult
from .pathology import PathologyAgent, PathologyAnalysis, TissueType

__all__ = [
    # Core agents
    "IntakeAgent",
    "ImagingAgent",
    "ReasoningAgent",
    "GuidelinesAgent",
    "PrimaCareOrchestrator",
    # New feature agents
    "VolumetricImagingAgent",
    "EHRNavigatorAgent",
    "PathologyAgent",
    # Data classes
    "LongitudinalAnalysis",
    "ChangeStatus",
    "VolumetricAnalysis",
    "VolumetricModality",
    "EHRQueryResult",
    "PathologyAnalysis",
    "TissueType",
]
