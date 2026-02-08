"""
Patient Education Agent - Health Literacy Translation

Converts technical clinical reports into patient-friendly language
at multiple reading levels. Designed for the Novel Task Prize track
in the MedGemma Impact Challenge.

Addresses the health literacy gap: ~36% of US adults have limited
health literacy (NAAL), leading to worse outcomes and higher costs.

Three reading levels:
- basic: 6th-grade reading level, simple terms only
- intermediate: common medical terms with explanations
- detailed: full clinical detail with terminology defined
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class PatientEducation:
    """Patient-friendly education output from a clinical report."""
    reading_level: str  # "basic", "intermediate", "detailed"
    simplified_diagnosis: str = ""
    what_it_means: str = ""
    next_steps: str = ""
    when_to_seek_help: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)
    raw_response: str = ""

    def to_prompt_context(self) -> str:
        """Format for inclusion in downstream prompts."""
        lines = [
            f"**Reading Level:** {self.reading_level}",
            f"**Simplified Diagnosis:** {self.simplified_diagnosis}",
            f"**What It Means:** {self.what_it_means}",
            f"**Next Steps:** {self.next_steps}",
            f"**When to Seek Help:** {self.when_to_seek_help}",
        ]
        if self.glossary:
            lines.append("\n**Glossary:**")
            for term, definition in self.glossary.items():
                lines.append(f"- {term}: {definition}")
        return "\n".join(lines)

    def to_report_section(self) -> str:
        """Generate formatted section for clinical report."""
        lines = [
            "-" * 40,
            f"PATIENT EDUCATION ({self.reading_level.upper()} LEVEL)",
            "-" * 40,
        ]
        if self.simplified_diagnosis:
            lines.append(f"\n**Your Diagnosis:**\n{self.simplified_diagnosis}")
        if self.what_it_means:
            lines.append(f"\n**What This Means:**\n{self.what_it_means}")
        if self.next_steps:
            lines.append(f"\n**What Happens Next:**\n{self.next_steps}")
        if self.when_to_seek_help:
            lines.append(f"\n**When to Get Help Right Away:**\n{self.when_to_seek_help}")
        if self.glossary:
            lines.append("\n**Medical Terms Explained:**")
            for term, definition in self.glossary.items():
                lines.append(f"- **{term}**: {definition}")
        return "\n".join(lines)


class PatientEducationAgent:
    """
    Agent that translates clinical reports into patient-friendly language.

    Supports three reading levels to match diverse health literacy needs.

    Usage:
        agent = PatientEducationAgent(model=medgemma)
        education = agent.educate(result, reading_level="basic")
        print(education.to_report_section())
    """

    BASIC_PROMPT = """You are a patient educator. Convert the following clinical report into language a 6th grader would understand. Use only simple, everyday words. Avoid all medical jargon.

## Clinical Report
{report_text}

---

Provide your response in this exact format:

**SIMPLIFIED DIAGNOSIS:**
[Explain what the doctor found, using only simple words]

**WHAT IT MEANS:**
[Explain what this means for the patient's health in simple terms]

**NEXT STEPS:**
[List what the patient needs to do next, step by step]

**WHEN TO SEEK HELP:**
[List warning signs that mean the patient should go to the doctor or hospital right away]

**GLOSSARY:**
- [medical term]: [simple definition]
- [medical term]: [simple definition]"""

    INTERMEDIATE_PROMPT = """You are a patient educator. Convert the following clinical report into language suitable for a general adult audience. Use common medical terms but explain them clearly.

## Clinical Report
{report_text}

---

Provide your response in this exact format:

**SIMPLIFIED DIAGNOSIS:**
[Explain the diagnosis using common medical terms with brief explanations]

**WHAT IT MEANS:**
[Explain the clinical significance in accessible language]

**NEXT STEPS:**
[List recommended follow-up actions and what to expect]

**WHEN TO SEEK HELP:**
[List specific symptoms or changes that warrant immediate medical attention]

**GLOSSARY:**
- [medical term]: [clear definition]
- [medical term]: [clear definition]"""

    DETAILED_PROMPT = """You are a patient educator. Convert the following clinical report into a detailed patient education document. Include clinical terminology but define all terms. This is for patients who want thorough understanding of their condition.

## Clinical Report
{report_text}

---

Provide your response in this exact format:

**SIMPLIFIED DIAGNOSIS:**
[Provide detailed clinical diagnosis with terminology explained inline]

**WHAT IT MEANS:**
[Explain pathophysiology and clinical implications in detail, defining terms]

**NEXT STEPS:**
[Detailed explanation of each recommended test, treatment, or follow-up and why it's needed]

**WHEN TO SEEK HELP:**
[Comprehensive list of warning signs with clinical context for each]

**GLOSSARY:**
- [medical term]: [detailed definition with clinical context]
- [medical term]: [detailed definition with clinical context]"""

    LEVEL_PROMPTS = {
        "basic": BASIC_PROMPT,
        "intermediate": INTERMEDIATE_PROMPT,
        "detailed": DETAILED_PROMPT,
    }

    def __init__(self, model=None):
        """
        Initialize the Patient Education Agent.

        Args:
            model: Optional MedGemma model instance
        """
        self._model = model

    @property
    def model(self):
        """Lazy load MedGemma model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def educate(self, result, reading_level: str = "basic") -> PatientEducation:
        """
        Generate patient-friendly education from a clinical result.

        Args:
            result: PrimaCareResult with clinical assessment
            reading_level: One of "basic", "intermediate", "detailed"

        Returns:
            PatientEducation with simplified content and glossary
        """
        level = reading_level.strip().lower()
        if level not in self.LEVEL_PROMPTS:
            raise ValueError(
                f"Unsupported reading_level='{reading_level}'. "
                f"Use one of: {', '.join(self.LEVEL_PROMPTS.keys())}"
            )

        # Get report text from result
        report_text = result.to_report()

        # Build prompt
        prompt_template = self.LEVEL_PROMPTS[level]
        prompt = prompt_template.format(report_text=report_text)

        # Call model
        response = self.model.ask(prompt, max_new_tokens=1500)

        # Parse response
        return self._parse_education(response, level)

    def educate_all_levels(self, result) -> Dict[str, PatientEducation]:
        """
        Generate patient education at all three reading levels.

        Args:
            result: PrimaCareResult with clinical assessment

        Returns:
            Dict mapping reading level to PatientEducation
        """
        return {
            level: self.educate(result, reading_level=level)
            for level in self.LEVEL_PROMPTS
        }

    def _parse_education(self, response: str, reading_level: str) -> PatientEducation:
        """Parse model response into PatientEducation dataclass."""
        sections = self._extract_sections(response)

        glossary = self._parse_glossary(sections.get("GLOSSARY", ""))

        return PatientEducation(
            reading_level=reading_level,
            simplified_diagnosis=sections.get("SIMPLIFIED DIAGNOSIS", "").strip(),
            what_it_means=sections.get("WHAT IT MEANS", "").strip(),
            next_steps=sections.get("NEXT STEPS", "").strip(),
            when_to_seek_help=sections.get("WHEN TO SEEK HELP", "").strip(),
            glossary=glossary,
            raw_response=response,
        )

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract labeled sections from response."""
        sections = {}
        current_section = None
        current_content: List[str] = []

        for line in text.split("\n"):
            # Check for section header like **SECTION NAME:**
            stripped = line.strip()
            if stripped.startswith("**") and stripped.endswith("**"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = stripped.strip("*:").strip().upper()
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _parse_glossary(self, text: str) -> Dict[str, str]:
        """Parse glossary entries from '- term: definition' format."""
        glossary = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match lines like "- term: definition" or "- **term**: definition"
            if line.startswith("-") or line.startswith("*"):
                content = line.lstrip("-*").strip()
                if ":" in content:
                    term, definition = content.split(":", 1)
                    term = term.strip().strip("*").strip()
                    definition = definition.strip()
                    if term and definition:
                        glossary[term] = definition
        return glossary
