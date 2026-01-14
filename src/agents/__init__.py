"""
PrimaCare AI - Agentic Workflow Components

Multi-agent system for primary care diagnostic support.
Designed for the MedGemma Impact Challenge - Agentic Workflow Prize.

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Intake    │ --> │   Imaging   │ --> │  Reasoning  │
    │   Agent     │     │   Agent     │     │   Agent     │
    └─────────────┘     └─────────────┘     └─────────────┘
         │                    │                   │
         v                    v                   v
    Structured HPI      Image Analysis      Differential Dx
                                            + Recommendations

Agents:
- IntakeAgent: Structures patient history into HPI format
- ImagingAgent: Analyzes chest X-rays with MedGemma
- ReasoningAgent: Combines clinical + imaging for diagnosis
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
from .imaging import ImagingAgent
from .reasoning import ReasoningAgent
from .orchestrator import PrimaCareOrchestrator

__all__ = [
    "IntakeAgent",
    "ImagingAgent",
    "ReasoningAgent",
    "PrimaCareOrchestrator",
]
