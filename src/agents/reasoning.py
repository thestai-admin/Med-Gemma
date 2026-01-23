"""
Reasoning Agent - Clinical Decision Support

Combines clinical history and imaging findings to generate:
- Differential diagnosis
- Recommended workup
- Risk stratification
- Clinical recommendations

This is the "brain" of the PrimaCare AI system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .intake import PatientContext, StructuredHPI, Urgency
from .imaging import ImageAnalysis


class WorkupPriority(Enum):
    """Priority levels for recommended workup."""
    STAT = "stat"  # Immediate
    URGENT = "urgent"  # Same day
    ROUTINE = "routine"  # Outpatient
    OPTIONAL = "optional"  # If clinically indicated


@dataclass
class Diagnosis:
    """A diagnosis with supporting evidence."""
    name: str
    probability: str  # "high", "moderate", "low"
    supporting_findings: List[str] = field(default_factory=list)
    against_findings: List[str] = field(default_factory=list)


@dataclass
class WorkupItem:
    """A recommended test or study."""
    name: str
    rationale: str
    priority: WorkupPriority = WorkupPriority.ROUTINE


@dataclass
class ClinicalRecommendation:
    """Complete clinical recommendation from reasoning."""
    differential_diagnosis: List[Diagnosis] = field(default_factory=list)
    most_likely_diagnosis: Optional[str] = None
    workup_recommendations: List[WorkupItem] = field(default_factory=list)
    disposition: str = ""  # "outpatient", "urgent care", "ED", "admit"
    follow_up: str = ""
    patient_instructions: str = ""
    red_flags_to_return: List[str] = field(default_factory=list)
    clinical_pearls: List[str] = field(default_factory=list)
    urgency: Urgency = Urgency.ROUTINE
    raw_reasoning: str = ""

    def to_summary(self) -> str:
        """Generate concise summary for display."""
        lines = []

        if self.most_likely_diagnosis:
            lines.append(f"**Most Likely:** {self.most_likely_diagnosis}")

        if self.differential_diagnosis:
            lines.append("\n**Differential Diagnosis:**")
            for i, dx in enumerate(self.differential_diagnosis[:5], 1):
                lines.append(f"{i}. {dx.name} ({dx.probability} probability)")

        if self.workup_recommendations:
            lines.append("\n**Recommended Workup:**")
            for item in self.workup_recommendations[:5]:
                priority = f"[{item.priority.value}]" if item.priority != WorkupPriority.ROUTINE else ""
                lines.append(f"- {item.name} {priority}")

        if self.disposition:
            lines.append(f"\n**Disposition:** {self.disposition}")

        if self.red_flags_to_return:
            lines.append("\n**Return Precautions:**")
            for flag in self.red_flags_to_return:
                lines.append(f"- {flag}")

        return "\n".join(lines)


class ReasoningAgent:
    """
    Agent for clinical reasoning and decision support.

    Combines patient history and imaging findings to generate
    differential diagnosis and treatment recommendations.

    Usage:
        agent = ReasoningAgent(medgemma_model)
        recommendation = agent.reason(
            patient_context=patient_context,
            imaging_analysis=imaging_analysis
        )
    """

    # Main reasoning prompt
    REASONING_PROMPT = """You are an experienced primary care physician assistant. Based on the following clinical information and imaging findings, provide a comprehensive clinical assessment.

## Clinical Information
{clinical_context}

## Imaging Findings
{imaging_context}

## Classification Results
{classification_context}

---

Please provide:

**MOST LIKELY DIAGNOSIS:**
[Single most likely diagnosis based on all available information]

**DIFFERENTIAL DIAGNOSIS:**
List the top 5 diagnoses in order of likelihood:
1. [Diagnosis] - [Probability: High/Moderate/Low]
   Supporting: [findings that support]
   Against: [findings that argue against]
2. ...

**RECOMMENDED WORKUP:**
List tests/studies needed to confirm diagnosis:
- [Test name] - [Rationale] - [Priority: STAT/URGENT/ROUTINE]

**DISPOSITION:**
[Outpatient / Urgent Care / Emergency Department / Hospital Admission]
Rationale: [why this disposition]

**FOLLOW-UP:**
[When and with whom patient should follow up]

**PATIENT INSTRUCTIONS:**
[Key instructions for the patient in plain language]

**RED FLAGS TO RETURN:**
List symptoms that should prompt immediate return:
- [symptom 1]
- [symptom 2]

**CLINICAL PEARLS:**
[Any important clinical considerations]"""

    # Prompt for differential diagnosis only
    DDX_PROMPT = """Based on this clinical scenario and imaging findings, provide a differential diagnosis.

Clinical Information:
{clinical_context}

Imaging Findings:
{imaging_context}

List the top 5 most likely diagnoses in order of probability.
For each diagnosis, explain the supporting and contradicting findings."""

    # Prompt for workup recommendations
    WORKUP_PROMPT = """Recommend the appropriate diagnostic workup for this patient.

Clinical Information:
{clinical_context}

Imaging Findings:
{imaging_context}

Working Diagnosis: {diagnosis}

List the recommended tests and studies:
1. [Test] - [Rationale] - [Priority]

Consider:
- Laboratory tests
- Additional imaging
- Specialist consultations
- Procedures"""

    def __init__(self, model=None):
        """
        Initialize the Reasoning Agent.

        Args:
            model: Optional MedGemma model instance
        """
        self._model = model

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def reason(
        self,
        patient_context: Optional[PatientContext] = None,
        imaging_analysis: Optional[ImageAnalysis] = None,
        clinical_text: Optional[str] = None,
        imaging_text: Optional[str] = None,
    ) -> ClinicalRecommendation:
        """
        Perform clinical reasoning on available information.

        Args:
            patient_context: Structured patient information
            imaging_analysis: Structured imaging results
            clinical_text: Alternative: raw clinical text
            imaging_text: Alternative: raw imaging text

        Returns:
            ClinicalRecommendation with differential and workup
        """
        # Build context strings
        if patient_context:
            clinical_context = patient_context.to_prompt_context()
        elif clinical_text:
            clinical_context = clinical_text
        else:
            clinical_context = "No clinical information provided."

        if imaging_analysis:
            imaging_context = imaging_analysis.to_prompt_context()
            classification_context = self._format_classifications(imaging_analysis)
        elif imaging_text:
            imaging_context = imaging_text
            classification_context = "No classification data available."
        else:
            imaging_context = "No imaging performed."
            classification_context = "N/A"

        # Run reasoning
        prompt = self.REASONING_PROMPT.format(
            clinical_context=clinical_context,
            imaging_context=imaging_context,
            classification_context=classification_context,
        )

        response = self.model.ask(prompt, max_new_tokens=1500)

        # Parse response
        recommendation = self._parse_recommendation(response)

        # Determine urgency
        recommendation.urgency = self._determine_urgency(
            patient_context, imaging_analysis, recommendation
        )

        return recommendation

    def get_differential(
        self,
        patient_context: Optional[PatientContext] = None,
        imaging_analysis: Optional[ImageAnalysis] = None,
        clinical_text: Optional[str] = None,
        imaging_text: Optional[str] = None,
    ) -> List[Diagnosis]:
        """
        Get just the differential diagnosis.

        Returns:
            List of Diagnosis objects
        """
        clinical_context = patient_context.to_prompt_context() if patient_context else (clinical_text or "")
        imaging_context = imaging_analysis.to_prompt_context() if imaging_analysis else (imaging_text or "")

        prompt = self.DDX_PROMPT.format(
            clinical_context=clinical_context,
            imaging_context=imaging_context,
        )

        response = self.model.ask(prompt, max_new_tokens=1500)
        return self._parse_differential(response)

    def get_workup(
        self,
        diagnosis: str,
        patient_context: Optional[PatientContext] = None,
        imaging_analysis: Optional[ImageAnalysis] = None,
    ) -> List[WorkupItem]:
        """
        Get workup recommendations for a specific diagnosis.

        Args:
            diagnosis: Working diagnosis
            patient_context: Patient information
            imaging_analysis: Imaging results

        Returns:
            List of WorkupItem recommendations
        """
        clinical_context = patient_context.to_prompt_context() if patient_context else ""
        imaging_context = imaging_analysis.to_prompt_context() if imaging_analysis else ""

        prompt = self.WORKUP_PROMPT.format(
            clinical_context=clinical_context,
            imaging_context=imaging_context,
            diagnosis=diagnosis,
        )

        response = self.model.ask(prompt, max_new_tokens=1000)
        return self._parse_workup(response)

    def explain_to_patient(
        self,
        recommendation: ClinicalRecommendation,
    ) -> str:
        """
        Generate patient-friendly explanation of findings.

        Args:
            recommendation: Clinical recommendation to explain

        Returns:
            Patient-friendly explanation string
        """
        prompt = f"""Explain the following clinical assessment to a patient in simple, non-medical terms.

Assessment:
{recommendation.to_summary()}

Please provide:
1. What we found (in simple terms)
2. What this means for you
3. What we need to do next
4. When to call or come back

Use empathetic, reassuring language while being honest about the findings."""

        return self.model.ask(prompt, max_new_tokens=1500)

    def _format_classifications(self, analysis: ImageAnalysis) -> str:
        """Format classification probabilities for prompt."""
        if not analysis.classification_probs:
            return "No classification data available."

        top = analysis.get_top_classifications(5)
        lines = ["Top classifications:"]
        for label, prob in top:
            lines.append(f"- {label}: {prob*100:.1f}%")
        return "\n".join(lines)

    def _parse_recommendation(self, response: str) -> ClinicalRecommendation:
        """Parse model response into ClinicalRecommendation."""
        rec = ClinicalRecommendation(raw_reasoning=response)

        sections = self._extract_sections(response)

        # Most likely diagnosis
        rec.most_likely_diagnosis = sections.get("MOST LIKELY DIAGNOSIS", "").strip()

        # Differential diagnosis
        ddx_text = sections.get("DIFFERENTIAL DIAGNOSIS", "")
        rec.differential_diagnosis = self._parse_differential(ddx_text)

        # Workup
        workup_text = sections.get("RECOMMENDED WORKUP", "")
        rec.workup_recommendations = self._parse_workup(workup_text)

        # Disposition
        rec.disposition = sections.get("DISPOSITION", "").split("\n")[0].strip()

        # Follow-up
        rec.follow_up = sections.get("FOLLOW-UP", "").strip()

        # Patient instructions
        rec.patient_instructions = sections.get("PATIENT INSTRUCTIONS", "").strip()

        # Red flags
        red_flags_text = sections.get("RED FLAGS TO RETURN", "")
        rec.red_flags_to_return = self._parse_list(red_flags_text)

        # Clinical pearls
        pearls_text = sections.get("CLINICAL PEARLS", "")
        rec.clinical_pearls = self._parse_list(pearls_text)

        return rec

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract labeled sections from response."""
        sections = {}
        current_section = None
        current_content = []

        for line in text.split("\n"):
            # Check for section header
            if line.strip().startswith("**") and line.strip().endswith("**"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.strip().strip("*:").strip().upper()
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _parse_differential(self, text: str) -> List[Diagnosis]:
        """Parse differential diagnosis text."""
        diagnoses = []
        current_dx = None

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for numbered diagnosis
            if line[0].isdigit() and "." in line:
                if current_dx:
                    diagnoses.append(current_dx)

                # Parse: "1. Pneumonia - High probability"
                content = line.split(".", 1)[1].strip()
                parts = content.split("-")
                name = parts[0].strip()
                prob = "moderate"
                if len(parts) > 1:
                    prob_text = parts[1].lower()
                    if "high" in prob_text:
                        prob = "high"
                    elif "low" in prob_text:
                        prob = "low"

                current_dx = Diagnosis(name=name, probability=prob)

            elif current_dx:
                # Parse supporting/against
                lower = line.lower()
                if "supporting" in lower or "support" in lower:
                    findings = line.split(":", 1)[-1].strip()
                    current_dx.supporting_findings = [f.strip() for f in findings.split(",")]
                elif "against" in lower:
                    findings = line.split(":", 1)[-1].strip()
                    current_dx.against_findings = [f.strip() for f in findings.split(",")]

        if current_dx:
            diagnoses.append(current_dx)

        return diagnoses

    def _parse_workup(self, text: str) -> List[WorkupItem]:
        """Parse workup recommendations."""
        items = []

        for line in text.split("\n"):
            line = line.strip()
            if not line or not (line[0].isdigit() or line.startswith("-")):
                continue

            content = line.lstrip("0123456789.-) ").strip()
            if not content:
                continue

            # Parse: "CBC - to evaluate infection - URGENT"
            parts = content.split("-")
            name = parts[0].strip()
            rationale = parts[1].strip() if len(parts) > 1 else ""

            priority = WorkupPriority.ROUTINE
            if len(parts) > 2:
                prio_text = parts[-1].strip().upper()
                if "STAT" in prio_text:
                    priority = WorkupPriority.STAT
                elif "URGENT" in prio_text:
                    priority = WorkupPriority.URGENT

            items.append(WorkupItem(
                name=name,
                rationale=rationale,
                priority=priority,
            ))

        return items

    def _parse_list(self, text: str) -> List[str]:
        """Parse bullet/numbered list."""
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                content = line.lstrip("-•0123456789.) ").strip()
                if content:
                    items.append(content)
        return items

    def _determine_urgency(
        self,
        patient_context: Optional[PatientContext],
        imaging_analysis: Optional[ImageAnalysis],
        recommendation: ClinicalRecommendation,
    ) -> Urgency:
        """Determine overall urgency based on all factors."""
        # Check patient context urgency
        if patient_context and patient_context.hpi.urgency in [Urgency.URGENT, Urgency.EMERGENT]:
            return patient_context.hpi.urgency

        # Check imaging urgency
        if imaging_analysis and imaging_analysis.requires_urgent_review:
            return Urgency.URGENT

        # Check disposition
        disposition = recommendation.disposition.lower()
        if "emergency" in disposition or "ed" in disposition:
            return Urgency.EMERGENT
        elif "admit" in disposition:
            return Urgency.URGENT
        elif "urgent" in disposition:
            return Urgency.SOON

        # Check for STAT workup items
        for item in recommendation.workup_recommendations:
            if item.priority == WorkupPriority.STAT:
                return Urgency.URGENT

        return Urgency.ROUTINE
