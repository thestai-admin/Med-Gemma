"""
PrimaCare AI - Agentic Workflow Components

Multi-agent system for primary care diagnostic support.
Designed for the MedGemma Impact Challenge.

Architecture:
    ┌─────────┐   ┌─────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐
    │ Intake  │-->│ Imaging │-->│ Reasoning │-->│ Guidelines │-->│ Education │
    │ Agent   │   │ Agent   │   │ Agent     │   │ Agent      │   │ Agent     │
    └─────────┘   └─────────┘   └───────────┘   └────────────┘   └───────────┘
        │              │             │                │                │
        v              v             v                v                v
    Structured     X-ray       Differential     Evidence-Based    Patient-Friendly
    HPI          Analysis      + Workup        Recommendations    Education

Agents:
- IntakeAgent: Structures patient history into HPI format
- ImagingAgent: Analyzes chest X-rays with MedGemma + MedSigLIP
- ReasoningAgent: Combines clinical + imaging for diagnosis
- GuidelinesAgent: RAG for clinical practice guidelines
- PatientEducationAgent: Translates reports into patient-friendly language
- PrimaCareOrchestrator: Coordinates all agents

Usage:
    from src.agents import PrimaCareOrchestrator

    orchestrator = PrimaCareOrchestrator()
    result = orchestrator.run(
        chief_complaint="Cough for 2 weeks",
        history="65yo male, smoker, fever",
        xray_image=image,
        include_education=True,
    )
"""

from .intake import IntakeAgent
from .imaging import ImagingAgent, LongitudinalAnalysis, ChangeStatus
from .reasoning import ReasoningAgent
from .guidelines import GuidelinesAgent
from .education import PatientEducationAgent
from .orchestrator import PrimaCareOrchestrator

__all__ = [
    # Core agents
    "IntakeAgent",
    "ImagingAgent",
    "ReasoningAgent",
    "GuidelinesAgent",
    "PatientEducationAgent",
    "PrimaCareOrchestrator",
    # Data classes
    "LongitudinalAnalysis",
    "ChangeStatus",
]
