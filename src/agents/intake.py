"""
Intake Agent - Patient History Structuring

Transforms unstructured patient information into a structured
History of Present Illness (HPI) format for clinical decision support.

Uses MedGemma's medical language understanding to:
- Extract key clinical elements from free text
- Organize into standard HPI components
- Identify relevant positives and negatives
- Flag critical symptoms requiring urgent attention
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class Urgency(Enum):
    """Urgency levels for clinical findings."""
    ROUTINE = "routine"
    SOON = "soon"  # Within days
    URGENT = "urgent"  # Within hours
    EMERGENT = "emergent"  # Immediate


@dataclass
class StructuredHPI:
    """Structured History of Present Illness."""
    chief_complaint: str
    onset: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    character: Optional[str] = None
    aggravating_factors: List[str] = field(default_factory=list)
    relieving_factors: List[str] = field(default_factory=list)
    timing: Optional[str] = None
    severity: Optional[str] = None
    associated_symptoms: List[str] = field(default_factory=list)
    pertinent_negatives: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    urgency: Urgency = Urgency.ROUTINE
    raw_input: str = ""

    def to_prompt_context(self) -> str:
        """Format HPI for use in downstream prompts."""
        lines = [f"**Chief Complaint:** {self.chief_complaint}"]

        if self.onset:
            lines.append(f"**Onset:** {self.onset}")
        if self.duration:
            lines.append(f"**Duration:** {self.duration}")
        if self.location:
            lines.append(f"**Location:** {self.location}")
        if self.character:
            lines.append(f"**Character:** {self.character}")
        if self.severity:
            lines.append(f"**Severity:** {self.severity}")
        if self.timing:
            lines.append(f"**Timing:** {self.timing}")
        if self.aggravating_factors:
            lines.append(f"**Aggravating Factors:** {', '.join(self.aggravating_factors)}")
        if self.relieving_factors:
            lines.append(f"**Relieving Factors:** {', '.join(self.relieving_factors)}")
        if self.associated_symptoms:
            lines.append(f"**Associated Symptoms:** {', '.join(self.associated_symptoms)}")
        if self.pertinent_negatives:
            lines.append(f"**Pertinent Negatives:** {', '.join(self.pertinent_negatives)}")
        if self.red_flags:
            lines.append(f"**Red Flags:** {', '.join(self.red_flags)}")

        return "\n".join(lines)


@dataclass
class PatientContext:
    """Complete patient context for analysis."""
    hpi: StructuredHPI
    age: Optional[int] = None
    gender: Optional[str] = None
    past_medical_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    social_history: Optional[str] = None
    family_history: Optional[str] = None

    def to_prompt_context(self) -> str:
        """Format full context for downstream prompts."""
        lines = []

        # Demographics
        demo_parts = []
        if self.age:
            demo_parts.append(f"{self.age} year old")
        if self.gender:
            demo_parts.append(self.gender)
        if demo_parts:
            lines.append(f"**Patient:** {' '.join(demo_parts)}")

        # HPI
        lines.append("\n**History of Present Illness:**")
        lines.append(self.hpi.to_prompt_context())

        # PMH
        if self.past_medical_history:
            lines.append(f"\n**Past Medical History:** {', '.join(self.past_medical_history)}")
        if self.medications:
            lines.append(f"**Medications:** {', '.join(self.medications)}")
        if self.allergies:
            lines.append(f"**Allergies:** {', '.join(self.allergies)}")
        if self.social_history:
            lines.append(f"**Social History:** {self.social_history}")
        if self.family_history:
            lines.append(f"**Family History:** {self.family_history}")

        return "\n".join(lines)


class IntakeAgent:
    """
    Agent for structuring patient intake information.

    Takes unstructured patient history and converts it to
    a structured format suitable for clinical decision support.

    Usage:
        agent = IntakeAgent(medgemma_model)
        hpi = agent.structure_history(
            chief_complaint="Cough for 2 weeks",
            history="65yo male smoker with fever and night sweats"
        )
    """

    # Prompt for structuring HPI
    STRUCTURE_PROMPT = """You are a medical intake assistant. Extract and structure the following patient information into a formal History of Present Illness (HPI).

Patient Information:
{input_text}

Please extract the following elements (if present):
1. Chief Complaint: The main reason for the visit
2. Onset: When did symptoms begin?
3. Location: Where is the symptom located?
4. Duration: How long have symptoms lasted?
5. Character: What does it feel like? (sharp, dull, etc.)
6. Aggravating Factors: What makes it worse?
7. Relieving Factors: What makes it better?
8. Timing: Pattern of symptoms (constant, intermittent, etc.)
9. Severity: How severe? (scale 1-10 or descriptive)
10. Associated Symptoms: Other symptoms present
11. Pertinent Negatives: Important symptoms that are NOT present
12. Red Flags: Any concerning features requiring urgent evaluation

Format your response as:
CHIEF_COMPLAINT: [main complaint]
ONSET: [when started]
LOCATION: [where]
DURATION: [how long]
CHARACTER: [description]
AGGRAVATING: [factors]
RELIEVING: [factors]
TIMING: [pattern]
SEVERITY: [level]
ASSOCIATED: [other symptoms]
NEGATIVES: [pertinent negatives]
RED_FLAGS: [concerning features]
URGENCY: [ROUTINE/SOON/URGENT/EMERGENT]"""

    # Red flag symptoms that suggest urgent evaluation
    RED_FLAG_KEYWORDS = [
        "chest pain", "shortness of breath", "hemoptysis", "syncope",
        "severe headache", "worst headache", "sudden onset", "fever",
        "weight loss", "night sweats", "hematemesis", "melena",
        "altered mental status", "weakness", "numbness", "vision loss",
        "suicidal", "homicidal"
    ]

    def __init__(self, model=None):
        """
        Initialize the Intake Agent.

        Args:
            model: Optional MedGemma model instance. If None, will be
                   initialized lazily when needed.
        """
        self._model = model

    @property
    def model(self):
        """Lazy load model if not provided."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def structure_history(
        self,
        chief_complaint: str,
        history: str = "",
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> StructuredHPI:
        """
        Structure patient history into HPI format.

        Args:
            chief_complaint: Main reason for visit
            history: Additional history text
            age: Patient age
            gender: Patient gender

        Returns:
            StructuredHPI with extracted elements
        """
        # Combine input
        input_parts = [f"Chief Complaint: {chief_complaint}"]
        if age:
            input_parts.append(f"Age: {age}")
        if gender:
            input_parts.append(f"Gender: {gender}")
        if history:
            input_parts.append(f"Additional History: {history}")

        input_text = "\n".join(input_parts)

        # Use model to structure
        prompt = self.STRUCTURE_PROMPT.format(input_text=input_text)
        response = self.model.ask(prompt, max_new_tokens=1000)

        # Parse response
        hpi = self._parse_structured_response(response, chief_complaint, input_text)

        # Check for red flags in original input
        self._check_red_flags(hpi, input_text.lower())

        return hpi

    def create_patient_context(
        self,
        chief_complaint: str,
        history: str = "",
        age: Optional[int] = None,
        gender: Optional[str] = None,
        pmh: Optional[List[str]] = None,
        medications: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        social_hx: Optional[str] = None,
        family_hx: Optional[str] = None,
    ) -> PatientContext:
        """
        Create full patient context including HPI and other history.

        Args:
            chief_complaint: Main reason for visit
            history: Additional history text
            age: Patient age
            gender: Patient gender
            pmh: Past medical history list
            medications: Current medications
            allergies: Known allergies
            social_hx: Social history
            family_hx: Family history

        Returns:
            PatientContext with all available information
        """
        hpi = self.structure_history(chief_complaint, history, age, gender)

        return PatientContext(
            hpi=hpi,
            age=age,
            gender=gender,
            past_medical_history=pmh or [],
            medications=medications or [],
            allergies=allergies or [],
            social_history=social_hx,
            family_history=family_hx,
        )

    def _parse_structured_response(
        self,
        response: str,
        chief_complaint: str,
        raw_input: str,
    ) -> StructuredHPI:
        """Parse model response into StructuredHPI."""
        hpi = StructuredHPI(
            chief_complaint=chief_complaint,
            raw_input=raw_input,
        )

        lines = response.strip().split("\n")
        for line in lines:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()

            if not value or value.lower() in ["none", "n/a", "not mentioned", "unknown"]:
                continue

            if key == "CHIEF_COMPLAINT":
                hpi.chief_complaint = value
            elif key == "ONSET":
                hpi.onset = value
            elif key == "LOCATION":
                hpi.location = value
            elif key == "DURATION":
                hpi.duration = value
            elif key == "CHARACTER":
                hpi.character = value
            elif key == "AGGRAVATING":
                hpi.aggravating_factors = [f.strip() for f in value.split(",")]
            elif key == "RELIEVING":
                hpi.relieving_factors = [f.strip() for f in value.split(",")]
            elif key == "TIMING":
                hpi.timing = value
            elif key == "SEVERITY":
                hpi.severity = value
            elif key == "ASSOCIATED":
                hpi.associated_symptoms = [s.strip() for s in value.split(",")]
            elif key == "NEGATIVES":
                hpi.pertinent_negatives = [n.strip() for n in value.split(",")]
            elif key == "RED_FLAGS":
                hpi.red_flags = [r.strip() for r in value.split(",")]
            elif key == "URGENCY":
                try:
                    hpi.urgency = Urgency(value.lower())
                except ValueError:
                    pass

        return hpi

    def _check_red_flags(self, hpi: StructuredHPI, text: str) -> None:
        """Check for red flag symptoms and update urgency."""
        found_flags = []
        for flag in self.RED_FLAG_KEYWORDS:
            if flag in text:
                found_flags.append(flag)

        if found_flags:
            # Add to red flags if not already present
            for flag in found_flags:
                if flag not in hpi.red_flags:
                    hpi.red_flags.append(flag)

            # Upgrade urgency if red flags found
            if hpi.urgency == Urgency.ROUTINE:
                hpi.urgency = Urgency.SOON
