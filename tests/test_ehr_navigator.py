"""
Tests for EHR/FHIR Navigator Agent

Tests the LangGraph workflow for navigating FHIR-formatted EHR data.
"""

import pytest
from unittest.mock import Mock, patch
import json

from src.agents.ehr_navigator import (
    EHRNavigatorAgent,
    EHRQueryResult,
    RetrievedFact,
    FHIRResource,
    FHIRResourceType,
)


@pytest.fixture
def mock_model():
    """Create a mock MedGemma model."""
    model = Mock()
    model.ask = Mock(return_value="""
RELEVANT_RESOURCES: Condition, MedicationRequest, Observation

**ANSWER:**
The patient has type 2 diabetes and hypertension as active conditions.

**CONFIDENCE:**
HIGH

**CAVEATS:**
None identified.
""")
    return model


@pytest.fixture
def sample_fhir_bundle():
    """Create a sample FHIR bundle for testing."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-1",
                    "name": [{"given": ["John"], "family": "Doe"}],
                    "birthDate": "1960-01-15",
                    "gender": "male",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-1",
                    "code": {
                        "coding": [{"display": "Type 2 Diabetes Mellitus"}]
                    },
                    "clinicalStatus": {
                        "coding": [{"code": "active"}]
                    },
                    "onsetDateTime": "2015-03-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-2",
                    "code": {
                        "coding": [{"display": "Essential Hypertension"}]
                    },
                    "clinicalStatus": {
                        "coding": [{"code": "active"}]
                    },
                    "onsetDateTime": "2010-06-15",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-1",
                    "medicationCodeableConcept": {
                        "coding": [{"display": "Metformin 500mg"}]
                    },
                    "status": "active",
                    "authoredOn": "2023-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "code": {
                        "coding": [{"display": "Blood Pressure"}]
                    },
                    "valueQuantity": {"value": 140, "unit": "mmHg"},
                    "effectiveDateTime": "2024-01-15",
                }
            },
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "allergy-1",
                    "code": {
                        "coding": [{"display": "Penicillin"}]
                    },
                    "criticality": "high",
                }
            },
        ],
    }


class TestEHRQueryResultDataclass:
    """Tests for EHRQueryResult dataclass."""

    def test_creation(self):
        """Test creating EHRQueryResult."""
        result = EHRQueryResult(
            question="What are the patient's conditions?",
            answer="The patient has diabetes and hypertension.",
            facts=[],
            resources_searched=["Condition"],
        )
        assert result.question == "What are the patient's conditions?"
        assert "diabetes" in result.answer

    def test_to_prompt_context(self):
        """Test formatting for downstream prompts."""
        result = EHRQueryResult(
            question="What medications?",
            answer="Metformin 500mg",
            facts=[
                RetrievedFact(
                    content="Taking Metformin 500mg",
                    source_resource_type="MedicationRequest",
                    source_resource_id="med-1",
                    date="2023-01-01",
                )
            ],
            confidence="high",
        )
        context = result.to_prompt_context()
        assert "Metformin" in context
        assert "MedicationRequest" in context


class TestRetrievedFact:
    """Tests for RetrievedFact dataclass."""

    def test_creation(self):
        """Test creating RetrievedFact."""
        fact = RetrievedFact(
            content="Patient has diabetes",
            source_resource_type="Condition",
            source_resource_id="condition-1",
            date="2015-03-01",
            relevance="Primary diagnosis",
        )
        assert fact.content == "Patient has diabetes"
        assert fact.source_resource_type == "Condition"


class TestEHRNavigatorAgent:
    """Tests for EHRNavigatorAgent."""

    def test_query_basic(self, mock_model, sample_fhir_bundle):
        """Test basic EHR query."""
        # Use without LangGraph for simpler testing
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        result = agent.query(
            question="What are the patient's active conditions?",
            fhir_bundle=sample_fhir_bundle,
        )

        assert isinstance(result, EHRQueryResult)
        assert result.question == "What are the patient's active conditions?"

    def test_simple_query_fallback(self, mock_model, sample_fhir_bundle):
        """Test simple query fallback when LangGraph unavailable."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        result = agent._simple_query(
            question="What medications is the patient taking?",
            fhir_bundle=sample_fhir_bundle,
        )

        assert isinstance(result, EHRQueryResult)
        assert "SIMPLE_QUERY" in result.workflow_trace[0]

    def test_get_patient_summary(self, mock_model, sample_fhir_bundle):
        """Test patient summary generation."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        summary = agent.get_patient_summary(sample_fhir_bundle)

        assert "PATIENT SUMMARY" in summary
        assert "John" in summary or "Doe" in summary
        assert "Diabetes" in summary or "Conditions" in summary

    def test_resource_summarization(self, mock_model):
        """Test summarization of different resource types."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        # Test Patient resource
        patient = {
            "resourceType": "Patient",
            "name": [{"given": ["Jane"], "family": "Smith"}],
            "birthDate": "1975-05-20",
            "gender": "female",
        }
        summary = agent._summarize_resource(patient)
        assert "Jane" in summary
        assert "1975-05-20" in summary

        # Test Condition resource
        condition = {
            "resourceType": "Condition",
            "code": {"coding": [{"display": "Asthma"}]},
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "onsetDateTime": "2020-01-01",
        }
        summary = agent._summarize_resource(condition)
        assert "Asthma" in summary

        # Test MedicationRequest resource
        med = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"coding": [{"display": "Lisinopril 10mg"}]},
            "status": "active",
            "authoredOn": "2024-01-01",
        }
        summary = agent._summarize_resource(med)
        assert "Lisinopril" in summary

    def test_load_fhir_bundle(self, mock_model, tmp_path):
        """Test loading FHIR bundle from file."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        # Create a test FHIR file
        test_bundle = {"resourceType": "Bundle", "entry": []}
        fhir_path = tmp_path / "test_bundle.json"
        with open(fhir_path, "w") as f:
            json.dump(test_bundle, f)

        loaded = agent.load_fhir_bundle(fhir_path)
        assert loaded["resourceType"] == "Bundle"


class TestFactParsing:
    """Tests for fact extraction and parsing."""

    def test_parse_facts(self, mock_model):
        """Test parsing facts from model response."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        response = """
FACT: Patient diagnosed with type 2 diabetes
DATE: 2015-03-01
RELEVANCE: Primary diagnosis

FACT: Currently on Metformin therapy
DATE: 2023-01-01
RELEVANCE: Active medication
"""

        facts = agent._parse_facts(response, "Condition", "cond-1")

        assert len(facts) == 2
        assert facts[0].content == "Patient diagnosed with type 2 diabetes"
        assert facts[0].date == "2015-03-01"
        assert facts[1].content == "Currently on Metformin therapy"


class TestWorkflowStages:
    """Tests for individual workflow stages."""

    def test_discover_node(self, mock_model, sample_fhir_bundle):
        """Test the discover stage."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        state = {
            "question": "Test question",
            "fhir_bundle": sample_fhir_bundle,
            "available_resources": [],
            "workflow_trace": [],
        }

        result = agent._discover_node(state)

        assert len(result["available_resources"]) > 0
        assert "Patient" in result["available_resources"]
        assert "Condition" in result["available_resources"]
        assert "DISCOVER" in result["workflow_trace"][0]

    def test_identify_node(self, mock_model, sample_fhir_bundle):
        """Test the identify stage."""
        agent = EHRNavigatorAgent(model=mock_model, use_langgraph=False)

        state = {
            "question": "What medications is the patient taking?",
            "fhir_bundle": sample_fhir_bundle,
            "available_resources": ["Patient", "Condition", "MedicationRequest"],
            "relevant_resource_types": [],
            "workflow_trace": [],
        }

        result = agent._identify_node(state)

        assert len(result["relevant_resource_types"]) > 0
        assert "IDENTIFY" in result["workflow_trace"][0]


class TestFHIRResourceType:
    """Tests for FHIRResourceType enum."""

    def test_common_resource_types(self):
        """Verify common FHIR resource types exist."""
        assert FHIRResourceType.PATIENT.value == "Patient"
        assert FHIRResourceType.CONDITION.value == "Condition"
        assert FHIRResourceType.OBSERVATION.value == "Observation"
        assert FHIRResourceType.MEDICATION_REQUEST.value == "MedicationRequest"
        assert FHIRResourceType.ALLERGY_INTOLERANCE.value == "AllergyIntolerance"
