"""
PrimaCare Orchestrator - Multi-Agent Coordinator

Orchestrates the Intake, Imaging, and Reasoning agents to provide
end-to-end diagnostic support for primary care.

This is the main entry point for the PrimaCare AI system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from PIL import Image
from datetime import datetime

from .intake import IntakeAgent, PatientContext, StructuredHPI, Urgency
from .imaging import ImagingAgent, ImageAnalysis, ImageModality
from .reasoning import ReasoningAgent, ClinicalRecommendation
from .guidelines import GuidelinesAgent, GuidelinesResult


@dataclass
class PrimaCareResult:
    """Complete result from PrimaCare AI analysis."""
    # Patient information
    patient_context: Optional[PatientContext] = None

    # Imaging results
    imaging_analysis: Optional[ImageAnalysis] = None

    # Clinical reasoning
    recommendation: Optional[ClinicalRecommendation] = None

    # Evidence-based guidelines
    guidelines_result: Optional[GuidelinesResult] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_steps: List[str] = field(default_factory=list)
    overall_urgency: Urgency = Urgency.ROUTINE

    def to_report(self) -> str:
        """Generate complete clinical report."""
        lines = [
            "=" * 60,
            "PRIMACARE AI - CLINICAL ASSESSMENT",
            "=" * 60,
            f"Generated: {self.timestamp}",
            f"Urgency: {self.overall_urgency.value.upper()}",
            "",
        ]

        # Patient Context
        if self.patient_context:
            lines.append("-" * 40)
            lines.append("PATIENT INFORMATION")
            lines.append("-" * 40)
            lines.append(self.patient_context.to_prompt_context())
            lines.append("")

        # Imaging
        if self.imaging_analysis:
            lines.append("-" * 40)
            lines.append("IMAGING ANALYSIS")
            lines.append("-" * 40)
            lines.append(self.imaging_analysis.to_prompt_context())
            lines.append("")

        # Clinical Recommendation
        if self.recommendation:
            lines.append("-" * 40)
            lines.append("CLINICAL ASSESSMENT")
            lines.append("-" * 40)
            lines.append(self.recommendation.to_summary())
            lines.append("")

        # Evidence-Based Guidelines
        if self.guidelines_result and self.guidelines_result.recommendations:
            lines.append(self.guidelines_result.to_report_section())
            lines.append("")

        lines.append("=" * 60)
        lines.append("DISCLAIMER: This is AI-generated content for clinical")
        lines.append("decision support only. All findings require verification")
        lines.append("by a qualified healthcare professional.")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "urgency": self.overall_urgency.value,
            "patient": {
                "chief_complaint": self.patient_context.hpi.chief_complaint if self.patient_context else None,
                "age": self.patient_context.age if self.patient_context else None,
                "gender": self.patient_context.gender if self.patient_context else None,
            },
            "imaging": {
                "modality": self.imaging_analysis.modality.value if self.imaging_analysis else None,
                "impression": self.imaging_analysis.impression if self.imaging_analysis else None,
                "urgent": self.imaging_analysis.requires_urgent_review if self.imaging_analysis else False,
            },
            "assessment": {
                "most_likely": self.recommendation.most_likely_diagnosis if self.recommendation else None,
                "differential": [d.name for d in (self.recommendation.differential_diagnosis[:5] if self.recommendation else [])],
                "disposition": self.recommendation.disposition if self.recommendation else None,
            },
            "guidelines": {
                "recommendations": [
                    {"text": r.recommendation, "evidence": r.evidence_level, "source": r.source_guidelines[0] if r.source_guidelines else None}
                    for r in (self.guidelines_result.recommendations if self.guidelines_result else [])
                ],
                "conditions_matched": self.guidelines_result.conditions_matched if self.guidelines_result else [],
            },
            "processing_steps": self.processing_steps,
        }


class PrimaCareOrchestrator:
    """
    Main orchestrator for PrimaCare AI diagnostic support.

    Coordinates three agents:
    1. IntakeAgent - Structures patient history
    2. ImagingAgent - Analyzes medical images
    3. ReasoningAgent - Generates differential and recommendations

    Usage:
        orchestrator = PrimaCareOrchestrator()

        # Full pipeline with image
        result = orchestrator.run(
            chief_complaint="Cough for 2 weeks",
            history="65yo male smoker with fever",
            xray_image=image
        )

        # Without image
        result = orchestrator.run(
            chief_complaint="Chest pain",
            history="45yo female, worse with exertion"
        )

        # Just imaging
        result = orchestrator.analyze_image(image, clinical_context="...")

    print(result.to_report())
    """

    def __init__(
        self,
        model=None,
        classifier=None,
        load_classifier: bool = True,
        enable_guidelines: bool = True,
        guidelines_path: Optional[str] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            model: Optional shared MedGemma model instance
            classifier: Optional MedSigLIP classifier instance
            load_classifier: Whether to load classifier for imaging
            enable_guidelines: Whether to enable guidelines RAG agent
            guidelines_path: Optional path to guidelines data directory
        """
        self._model = model
        self._classifier = classifier
        self._load_classifier = load_classifier
        self._enable_guidelines = enable_guidelines
        self._guidelines_path = guidelines_path

        # Initialize agents (they will share the model)
        self._intake_agent = None
        self._imaging_agent = None
        self._reasoning_agent = None
        self._guidelines_agent = None

    @property
    def model(self):
        """Lazy load shared model."""
        if self._model is None:
            from ..model import MedGemma
            print("Loading MedGemma model...")
            self._model = MedGemma()
            print("Model loaded!")
        return self._model

    @property
    def intake_agent(self) -> IntakeAgent:
        """Get or create intake agent."""
        if self._intake_agent is None:
            self._intake_agent = IntakeAgent(model=self.model)
        return self._intake_agent

    @property
    def imaging_agent(self) -> ImagingAgent:
        """Get or create imaging agent."""
        if self._imaging_agent is None:
            self._imaging_agent = ImagingAgent(
                model=self.model,
                classifier=self._classifier,
                load_classifier=self._load_classifier,
            )
        return self._imaging_agent

    @property
    def reasoning_agent(self) -> ReasoningAgent:
        """Get or create reasoning agent."""
        if self._reasoning_agent is None:
            self._reasoning_agent = ReasoningAgent(model=self.model)
        return self._reasoning_agent

    @property
    def guidelines_agent(self) -> GuidelinesAgent:
        """Get or create guidelines agent."""
        if self._guidelines_agent is None:
            if self._guidelines_path:
                from pathlib import Path
                base_path = Path(self._guidelines_path)
                self._guidelines_agent = GuidelinesAgent(
                    model=self.model,
                    embeddings_path=str(base_path / "embeddings.npz"),
                    chunks_path=str(base_path / "chunks.json"),
                )
            else:
                self._guidelines_agent = GuidelinesAgent(model=self.model)
        return self._guidelines_agent

    def run(
        self,
        chief_complaint: str,
        history: str = "",
        xray_image: Optional[Union[Image.Image, str, Path]] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        pmh: Optional[List[str]] = None,
        medications: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        include_classification: bool = True,
    ) -> PrimaCareResult:
        """
        Run the complete diagnostic support pipeline.

        Args:
            chief_complaint: Main reason for visit
            history: Additional clinical history
            xray_image: Optional chest X-ray image
            age: Patient age
            gender: Patient gender
            pmh: Past medical history
            medications: Current medications
            allergies: Known allergies
            include_classification: Run zero-shot classification

        Returns:
            PrimaCareResult with complete analysis
        """
        result = PrimaCareResult()

        # Step 1: Structure patient history
        print("Step 1: Processing patient information...")
        result.processing_steps.append("intake_started")

        patient_context = self.intake_agent.create_patient_context(
            chief_complaint=chief_complaint,
            history=history,
            age=age,
            gender=gender,
            pmh=pmh,
            medications=medications,
            allergies=allergies,
        )
        result.patient_context = patient_context
        result.processing_steps.append("intake_completed")

        # Step 2: Analyze imaging (if provided)
        if xray_image is not None:
            print("Step 2: Analyzing chest X-ray...")
            result.processing_steps.append("imaging_started")

            # Build clinical context for imaging
            clinical_context = f"{chief_complaint}. {history}"
            if age:
                clinical_context = f"{age}yo {gender or 'patient'}. {clinical_context}"

            imaging_analysis = self.imaging_agent.analyze(
                image=xray_image,
                clinical_context=clinical_context,
                include_classification=include_classification,
            )
            result.imaging_analysis = imaging_analysis
            result.processing_steps.append("imaging_completed")
        else:
            print("Step 2: No imaging provided, skipping...")
            result.processing_steps.append("imaging_skipped")

        # Step 3: Clinical reasoning
        print("Step 3: Generating clinical assessment...")
        result.processing_steps.append("reasoning_started")

        recommendation = self.reasoning_agent.reason(
            patient_context=patient_context,
            imaging_analysis=result.imaging_analysis,
        )
        result.recommendation = recommendation
        result.processing_steps.append("reasoning_completed")

        # Step 4: Evidence-based guidelines (if enabled)
        if self._enable_guidelines and result.recommendation:
            print("Step 4: Retrieving clinical guidelines...")
            result.processing_steps.append("guidelines_started")

            try:
                # Get top diagnoses for guideline lookup
                differential = [
                    d.name for d in result.recommendation.differential_diagnosis[:3]
                ]

                guidelines_result = self.guidelines_agent.get_recommendations(
                    differential_diagnosis=differential,
                    chief_complaint=chief_complaint,
                )
                result.guidelines_result = guidelines_result
                result.processing_steps.append("guidelines_completed")
            except Exception as e:
                print(f"Warning: Guidelines retrieval failed: {e}")
                result.processing_steps.append("guidelines_failed")
        else:
            result.processing_steps.append("guidelines_skipped")

        # Determine overall urgency
        result.overall_urgency = self._determine_overall_urgency(result)

        print("Analysis complete!")
        return result

    def analyze_image(
        self,
        image: Union[Image.Image, str, Path],
        clinical_context: Optional[str] = None,
        include_classification: bool = True,
    ) -> PrimaCareResult:
        """
        Quick image-only analysis.

        Args:
            image: Chest X-ray image
            clinical_context: Optional clinical information
            include_classification: Run classification

        Returns:
            PrimaCareResult with imaging analysis
        """
        result = PrimaCareResult()
        result.processing_steps.append("imaging_only_mode")

        # Analyze image
        print("Analyzing image...")
        imaging_analysis = self.imaging_agent.analyze(
            image=image,
            clinical_context=clinical_context,
            include_classification=include_classification,
        )
        result.imaging_analysis = imaging_analysis
        result.processing_steps.append("imaging_completed")

        # Generate basic recommendation from imaging alone
        print("Generating assessment...")
        recommendation = self.reasoning_agent.reason(
            imaging_analysis=imaging_analysis,
            clinical_text=clinical_context,
        )
        result.recommendation = recommendation
        result.processing_steps.append("reasoning_completed")

        result.overall_urgency = self._determine_overall_urgency(result)
        return result

    def get_differential(
        self,
        chief_complaint: str,
        history: str = "",
        xray_image: Optional[Union[Image.Image, str, Path]] = None,
    ) -> List[str]:
        """
        Quick differential diagnosis.

        Returns:
            List of diagnosis names in order of likelihood
        """
        result = self.run(
            chief_complaint=chief_complaint,
            history=history,
            xray_image=xray_image,
            include_classification=False,
        )

        if result.recommendation and result.recommendation.differential_diagnosis:
            return [d.name for d in result.recommendation.differential_diagnosis]
        return []

    def explain_to_patient(
        self,
        result: PrimaCareResult,
    ) -> str:
        """
        Generate patient-friendly explanation.

        Args:
            result: Previous analysis result

        Returns:
            Patient-friendly explanation text
        """
        if result.recommendation:
            return self.reasoning_agent.explain_to_patient(result.recommendation)
        return "No assessment available to explain."

    def _determine_overall_urgency(self, result: PrimaCareResult) -> Urgency:
        """Determine overall urgency from all sources."""
        urgencies = []

        if result.patient_context:
            urgencies.append(result.patient_context.hpi.urgency)

        if result.imaging_analysis and result.imaging_analysis.requires_urgent_review:
            urgencies.append(Urgency.URGENT)

        if result.recommendation:
            urgencies.append(result.recommendation.urgency)

        # Return highest urgency
        priority = {
            Urgency.ROUTINE: 0,
            Urgency.SOON: 1,
            Urgency.URGENT: 2,
            Urgency.EMERGENT: 3,
        }

        if urgencies:
            return max(urgencies, key=lambda u: priority[u])
        return Urgency.ROUTINE


# =============================================================================
# Quick Functions
# =============================================================================

def analyze_patient(
    chief_complaint: str,
    history: str = "",
    xray_image: Optional[Union[Image.Image, str, Path]] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> PrimaCareResult:
    """
    Quick function for complete patient analysis.

    Usage:
        result = analyze_patient(
            chief_complaint="Cough for 2 weeks",
            history="Smoker, fever, night sweats",
            xray_image=image,
            age=65,
            gender="male"
        )
        print(result.to_report())
    """
    orchestrator = PrimaCareOrchestrator()
    return orchestrator.run(
        chief_complaint=chief_complaint,
        history=history,
        xray_image=xray_image,
        age=age,
        gender=gender,
    )


def quick_xray_analysis(
    image: Union[Image.Image, str, Path],
    clinical_context: Optional[str] = None,
) -> str:
    """
    Quick X-ray analysis returning summary text.

    Usage:
        summary = quick_xray_analysis(image, "65yo with cough")
        print(summary)
    """
    orchestrator = PrimaCareOrchestrator()
    result = orchestrator.analyze_image(image, clinical_context)
    return result.to_report()
