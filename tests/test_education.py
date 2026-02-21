"""
Tests for PatientEducationAgent.

All tests use mock models - no GPU required.
"""

import pytest
from unittest.mock import Mock, patch

from src.agents.education import PatientEducationAgent, PatientEducation, flesch_kincaid_grade


MOCK_EDUCATION_RESPONSE = """**SIMPLIFIED DIAGNOSIS:**
The chest X-ray shows signs of a lung infection called pneumonia.

**WHAT IT MEANS:**
You have an infection in your lungs that is making it hard to breathe.

**NEXT STEPS:**
1. Take the antibiotics your doctor prescribed
2. Rest at home for the next few days
3. Drink plenty of fluids
4. Follow up with your doctor in one week

**WHEN TO SEEK HELP:**
Go to the emergency room if you have:
- Trouble breathing or chest pain
- High fever above 103F
- Coughing up blood

**GLOSSARY:**
- pneumonia: an infection that causes inflammation in the lungs
- consolidation: an area of the lung filled with fluid or infection
- infiltrate: abnormal substance in the lung tissue"""


@pytest.fixture
def mock_result():
    """Create a mock PrimaCareResult for testing."""
    result = Mock()
    result.to_report.return_value = "Mock clinical report with pneumonia findings."
    return result


@pytest.fixture
def education_agent(mock_medgemma):
    """Create a PatientEducationAgent with mock model."""
    mock_medgemma.ask.return_value = MOCK_EDUCATION_RESPONSE
    return PatientEducationAgent(model=mock_medgemma)


def test_educate_basic_level_returns_patient_education(education_agent, mock_result):
    """Test that educate() returns a populated PatientEducation at basic level."""
    education = education_agent.educate(mock_result, reading_level="basic")

    assert isinstance(education, PatientEducation)
    assert education.reading_level == "basic"
    assert "pneumonia" in education.simplified_diagnosis.lower()
    assert education.what_it_means != ""
    assert education.next_steps != ""
    assert education.when_to_seek_help != ""


def test_educate_parses_glossary_correctly(education_agent, mock_result):
    """Test that glossary entries are parsed from '- term: definition' format."""
    education = education_agent.educate(mock_result, reading_level="basic")

    assert isinstance(education.glossary, dict)
    assert len(education.glossary) >= 2
    assert "pneumonia" in education.glossary
    assert "lung" in education.glossary["pneumonia"].lower()


def test_educate_all_levels_returns_three_results(education_agent, mock_result):
    """Test that educate_all_levels() returns results for all 3 levels."""
    results = education_agent.educate_all_levels(mock_result)

    assert isinstance(results, dict)
    assert set(results.keys()) == {"basic", "intermediate", "detailed"}
    for level, education in results.items():
        assert isinstance(education, PatientEducation)
        assert education.reading_level == level


def test_educate_invalid_level_raises(education_agent, mock_result):
    """Test that invalid reading level raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported reading_level"):
        education_agent.educate(mock_result, reading_level="expert")


def test_education_to_report_section():
    """Test PatientEducation.to_report_section() formatting."""
    education = PatientEducation(
        reading_level="basic",
        simplified_diagnosis="You have a lung infection.",
        what_it_means="An infection is making it hard to breathe.",
        next_steps="Take antibiotics and rest.",
        when_to_seek_help="Go to ER if fever above 103F.",
        glossary={"pneumonia": "a lung infection"},
    )

    section = education.to_report_section()
    assert "PATIENT EDUCATION (BASIC LEVEL)" in section
    assert "Your Diagnosis" in section
    assert "lung infection" in section
    assert "pneumonia" in section


def test_education_to_prompt_context():
    """Test PatientEducation.to_prompt_context() formatting."""
    education = PatientEducation(
        reading_level="intermediate",
        simplified_diagnosis="Community-acquired pneumonia",
        what_it_means="Infection causing lung inflammation",
        next_steps="Complete antibiotic course",
        when_to_seek_help="Return if worsening dyspnea",
        glossary={"dyspnea": "difficulty breathing"},
    )

    context = education.to_prompt_context()
    assert "intermediate" in context.lower()
    assert "dyspnea" in context


def test_orchestrator_education_step_runs_when_enabled(mock_medgemma, mock_medsiglip):
    """Test that education step runs when include_education=True."""
    mock_medgemma.ask.return_value = MOCK_EDUCATION_RESPONSE
    mock_medgemma.analyze_image.return_value = "Mock imaging analysis."

    from src.agents.orchestrator import PrimaCareOrchestrator

    orchestrator = PrimaCareOrchestrator(
        model=mock_medgemma,
        classifier=mock_medsiglip,
        enable_guidelines=False,
    )

    result = orchestrator.run(
        chief_complaint="Cough",
        history="65yo male",
        include_education=True,
        education_level="basic",
    )

    assert "education_completed" in result.processing_steps
    assert result.patient_education is not None
    assert result.patient_education.reading_level == "basic"


def test_orchestrator_education_skipped_by_default(mock_medgemma, mock_medsiglip):
    """Test that education is skipped by default."""
    mock_medgemma.ask.return_value = "Mock response."
    mock_medgemma.analyze_image.return_value = "Mock imaging."

    from src.agents.orchestrator import PrimaCareOrchestrator

    orchestrator = PrimaCareOrchestrator(
        model=mock_medgemma,
        classifier=mock_medsiglip,
        enable_guidelines=False,
    )

    result = orchestrator.run(
        chief_complaint="Headache",
        history="30yo female",
    )

    assert "education_skipped" in result.processing_steps
    assert result.patient_education is None


def test_education_profile_timing(mock_medgemma, mock_medsiglip):
    """Test that education timing is captured in profile mode."""
    mock_medgemma.ask.return_value = MOCK_EDUCATION_RESPONSE
    mock_medgemma.analyze_image.return_value = "Mock imaging."

    from src.agents.orchestrator import PrimaCareOrchestrator

    orchestrator = PrimaCareOrchestrator(
        model=mock_medgemma,
        classifier=mock_medsiglip,
        enable_guidelines=False,
    )

    result = orchestrator.run(
        chief_complaint="Cough",
        history="65yo male",
        include_education=True,
        profile=True,
    )

    assert "education" in result.timings
    assert result.timings["education"] >= 0


def test_flesch_kincaid_grade_simple_text():
    """Simple short-word text should score at a low grade level."""
    simple = "You have a cold. Rest and drink water. See the doctor if worse."
    grade = flesch_kincaid_grade(simple)
    assert isinstance(grade, float)
    assert grade < 8  # simple text should be below 8th grade


def test_flesch_kincaid_grade_complex_text():
    """Medical jargon text should score higher than simple text."""
    simple = "You are sick. Rest and drink water."
    complex_text = (
        "The radiological examination demonstrates consolidative opacification "
        "in the right lower lobe with associated bronchial pneumogram, consistent "
        "with community-acquired pneumococcal pneumonia requiring antibiotic therapy."
    )
    assert flesch_kincaid_grade(complex_text) > flesch_kincaid_grade(simple)


def test_educate_populates_flesch_kincaid_grade(education_agent, mock_result):
    """educate() should populate flesch_kincaid_grade on the result."""
    education = education_agent.educate(mock_result, reading_level="basic")
    assert education.flesch_kincaid_grade is not None
    assert isinstance(education.flesch_kincaid_grade, float)


def test_report_section_includes_grade(education_agent, mock_result):
    """to_report_section() should include the FK grade when available."""
    education = education_agent.educate(mock_result, reading_level="basic")
    section = education.to_report_section()
    assert "Flesch-Kincaid Grade" in section


def test_education_in_to_dict(mock_medgemma, mock_medsiglip):
    """Test that education appears in to_dict() output."""
    mock_medgemma.ask.return_value = MOCK_EDUCATION_RESPONSE
    mock_medgemma.analyze_image.return_value = "Mock imaging."

    from src.agents.orchestrator import PrimaCareOrchestrator

    orchestrator = PrimaCareOrchestrator(
        model=mock_medgemma,
        classifier=mock_medsiglip,
        enable_guidelines=False,
    )

    result = orchestrator.run(
        chief_complaint="Cough",
        include_education=True,
    )

    d = result.to_dict()
    assert "education" in d
    assert d["education"]["reading_level"] == "basic"
