"""Tests for orchestrator profiling and fast mode behavior."""

from PIL import Image

from src.agents.intake import PatientContext, StructuredHPI, Urgency
from src.agents.imaging import ImageAnalysis, ImageModality
from src.agents.orchestrator import PrimaCareOrchestrator
from src.agents.reasoning import ClinicalRecommendation, Diagnosis


class StubIntakeAgent:
    """Simple deterministic intake agent for orchestration tests."""

    def create_patient_context(
        self,
        chief_complaint,
        history="",
        age=None,
        gender=None,
        pmh=None,
        medications=None,
        allergies=None,
    ):
        return PatientContext(
            hpi=StructuredHPI(chief_complaint=chief_complaint, urgency=Urgency.ROUTINE),
            age=age,
            gender=gender,
            past_medical_history=pmh or [],
            medications=medications or [],
            allergies=allergies or [],
        )


class StubImagingAgent:
    """Simple deterministic imaging agent with call tracking."""

    def __init__(self):
        self.calls = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return ImageAnalysis(
            modality=ImageModality.CHEST_XRAY,
            technique="PA",
            quality="Adequate",
            impression="No acute cardiopulmonary process",
        )


class StubReasoningAgent:
    """Simple deterministic reasoning agent."""

    def reason(self, patient_context=None, imaging_analysis=None, clinical_text=None, imaging_text=None):
        return ClinicalRecommendation(
            most_likely_diagnosis="Viral upper respiratory infection",
            differential_diagnosis=[
                Diagnosis(name="Viral upper respiratory infection", probability="high")
            ],
            disposition="outpatient",
            urgency=Urgency.ROUTINE,
        )


def _build_orchestrator():
    orchestrator = PrimaCareOrchestrator(enable_guidelines=False)
    orchestrator._intake_agent = StubIntakeAgent()
    orchestrator._imaging_agent = StubImagingAgent()
    orchestrator._reasoning_agent = StubReasoningAgent()
    return orchestrator


def test_run_records_profile_timings_and_total():
    orchestrator = _build_orchestrator()

    result = orchestrator.run(
        chief_complaint="cough",
        history="2 days",
        xray_image=Image.new("RGB", (64, 64)),
        profile=True,
        include_classification=True,
        classification_mode="binary",
    )

    assert "intake" in result.timings
    assert "imaging" in result.timings
    assert "reasoning" in result.timings
    assert "total" in result.timings
    assert result.timings["total"] >= result.timings["imaging"]


def test_fast_mode_switches_multilabel_to_binary():
    orchestrator = _build_orchestrator()

    orchestrator.run(
        chief_complaint="cough",
        xray_image=Image.new("RGB", (64, 64)),
        include_classification=True,
        classification_mode="multilabel",
        fast_mode=True,
        profile=True,
    )

    imaging_agent = orchestrator._imaging_agent
    assert imaging_agent.calls
    assert imaging_agent.calls[0]["classification_mode"] == "binary"


def test_analyze_image_profile_populates_timing_fields():
    orchestrator = _build_orchestrator()

    result = orchestrator.analyze_image(
        image=Image.new("RGB", (64, 64)),
        profile=True,
        include_classification=False,
    )

    assert set(["imaging", "reasoning", "total"]).issubset(result.timings.keys())
