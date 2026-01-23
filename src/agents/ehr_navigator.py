"""
EHR Navigator Agent - FHIR Data Navigation

Navigates FHIR-formatted EHR data using a LangGraph workflow to
answer clinical questions from patient records.

5-Stage Workflow (following Google's HAI-DEF pattern):
1. DISCOVER - Get manifest of available FHIR resources
2. IDENTIFY - Determine relevant resource types for the query
3. SELECT - Plan specific resource retrievals
4. EXECUTE - Retrieve and extract facts from resources
5. SYNTHESIZE - Combine facts into a coherent answer

Memory Impact: Minimal (~50MB for LangGraph)
All MedGemma calls are text-only (no images)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, TypedDict
from pathlib import Path
from enum import Enum
import json


# FHIR Resource Types commonly used in EHR
class FHIRResourceType(Enum):
    """Common FHIR resource types."""
    PATIENT = "Patient"
    CONDITION = "Condition"
    OBSERVATION = "Observation"
    MEDICATION_REQUEST = "MedicationRequest"
    MEDICATION_STATEMENT = "MedicationStatement"
    PROCEDURE = "Procedure"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    IMMUNIZATION = "Immunization"
    ENCOUNTER = "Encounter"
    CARE_PLAN = "CarePlan"
    DOCUMENT_REFERENCE = "DocumentReference"
    FAMILY_MEMBER_HISTORY = "FamilyMemberHistory"
    LAB_RESULT = "Observation"  # Labs are Observations with category


@dataclass
class FHIRResource:
    """Represents a single FHIR resource."""
    resource_type: str
    resource_id: str
    data: Dict[str, Any]
    summary: str = ""  # Human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resourceType": self.resource_type,
            "id": self.resource_id,
            "data": self.data,
            "summary": self.summary,
        }


@dataclass
class RetrievedFact:
    """A fact extracted from a FHIR resource."""
    content: str
    source_resource_type: str
    source_resource_id: str
    relevance: str = ""  # Why this fact is relevant
    date: Optional[str] = None  # Date associated with the fact


@dataclass
class EHRQueryResult:
    """Result of an EHR query."""
    question: str
    answer: str
    facts: List[RetrievedFact] = field(default_factory=list)
    resources_searched: List[str] = field(default_factory=list)
    confidence: str = "medium"  # low, medium, high
    workflow_trace: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format result for downstream prompts."""
        lines = [
            f"**Question:** {self.question}",
            f"**Answer:** {self.answer}",
            f"**Confidence:** {self.confidence}",
        ]

        if self.facts:
            lines.append("")
            lines.append("**Supporting Facts:**")
            for i, fact in enumerate(self.facts, 1):
                date_str = f" ({fact.date})" if fact.date else ""
                lines.append(f"{i}. {fact.content}{date_str}")
                lines.append(f"   Source: {fact.source_resource_type}/{fact.source_resource_id}")

        return "\n".join(lines)

    def to_report_section(self) -> str:
        """Generate a report section."""
        return f"""
----------------------------------------
EHR QUERY RESULT
----------------------------------------
Question: {self.question}

Answer: {self.answer}

Confidence: {self.confidence}
Resources Searched: {', '.join(self.resources_searched)}
----------------------------------------
"""


# LangGraph State Definition
class AgentState(TypedDict):
    """State for the EHR Navigator workflow."""
    question: str
    fhir_bundle: Dict[str, Any]
    available_resources: List[str]
    relevant_resource_types: List[str]
    selected_resources: List[Dict[str, Any]]
    extracted_facts: List[Dict[str, Any]]
    answer: str
    workflow_trace: List[str]


class EHRNavigatorAgent:
    """
    Agent for navigating FHIR-formatted EHR data.

    Uses a LangGraph workflow to systematically search and
    synthesize information from patient records.

    Usage:
        agent = EHRNavigatorAgent()
        result = agent.query(
            question="What medications is the patient taking?",
            fhir_bundle=patient_fhir_data
        )
    """

    # Prompts for each workflow stage
    DISCOVER_PROMPT = """Analyze this FHIR bundle manifest and list all available resource types.

**Available Resources in Bundle:**
{manifest}

List each resource type and count. Format:
RESOURCE_TYPES: [comma-separated list]
TOTAL_RESOURCES: [number]
"""

    IDENTIFY_PROMPT = """Given a clinical question and available FHIR resources, identify which resource types are relevant.

**Question:** {question}

**Available Resource Types:** {available_resources}

Which FHIR resource types should be queried to answer this question?
Consider:
- Patient: Demographics, identifiers
- Condition: Diagnoses, problems
- Observation: Vitals, labs, assessments
- MedicationRequest/MedicationStatement: Medications
- Procedure: Surgeries, interventions
- DiagnosticReport: Imaging, pathology reports
- AllergyIntolerance: Allergies, adverse reactions
- Encounter: Visits, admissions
- FamilyMemberHistory: Family history

RELEVANT_RESOURCES: [comma-separated list in order of relevance]
REASONING: [brief explanation]
"""

    SELECT_PROMPT = """Plan which specific resources to retrieve for the query.

**Question:** {question}

**Relevant Resource Types:** {relevant_types}

**Resources Available:**
{resources_summary}

Select the most relevant specific resources to answer the question.
Consider date relevance (recent records often more relevant).

SELECTED_RESOURCES: [list resource IDs to retrieve]
PRIORITY_ORDER: [order of importance]
"""

    EXTRACT_PROMPT = """Extract relevant facts from this FHIR resource to answer the clinical question.

**Question:** {question}

**Resource Type:** {resource_type}
**Resource Data:**
{resource_data}

Extract all facts relevant to answering the question.
Format each fact as:
FACT: [the fact]
DATE: [date if available]
RELEVANCE: [why relevant to question]

If no relevant facts, respond with: NO_RELEVANT_FACTS
"""

    SYNTHESIZE_PROMPT = """Synthesize facts from the patient's EHR to answer the clinical question.

**Question:** {question}

**Extracted Facts:**
{facts}

**Resource Types Searched:** {searched_resources}

Provide a comprehensive answer:

**ANSWER:**
[Detailed answer to the question based on the facts]

**CONFIDENCE:**
[HIGH/MEDIUM/LOW] - How confident are you in this answer based on available data?

**CAVEATS:**
[Any limitations or missing information that affects the answer]
"""

    def __init__(self, model=None, use_langgraph: bool = True):
        """
        Initialize the EHR Navigator Agent.

        Args:
            model: Optional MedGemma model instance
            use_langgraph: Whether to use LangGraph (requires langgraph package)
        """
        self._model = model
        self._use_langgraph = use_langgraph
        self._workflow = None

        if use_langgraph:
            self._build_workflow()

    @property
    def model(self):
        """Lazy load MedGemma model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            print("Warning: langgraph not installed. Using fallback workflow.")
            self._use_langgraph = False
            return

        workflow = StateGraph(AgentState)

        # Add nodes for each stage
        workflow.add_node("discover", self._discover_node)
        workflow.add_node("identify", self._identify_node)
        workflow.add_node("select", self._select_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges (linear flow)
        workflow.set_entry_point("discover")
        workflow.add_edge("discover", "identify")
        workflow.add_edge("identify", "select")
        workflow.add_edge("select", "execute")
        workflow.add_edge("execute", "synthesize")
        workflow.add_edge("synthesize", END)

        self._workflow = workflow.compile()

    def _discover_node(self, state: AgentState) -> AgentState:
        """Discover available resources in the FHIR bundle."""
        state["workflow_trace"].append("DISCOVER: Analyzing FHIR bundle manifest")

        bundle = state["fhir_bundle"]

        # Extract resource manifest
        resource_counts = {}
        if "entry" in bundle:
            for entry in bundle["entry"]:
                resource = entry.get("resource", {})
                rtype = resource.get("resourceType", "Unknown")
                resource_counts[rtype] = resource_counts.get(rtype, 0) + 1

        manifest = "\n".join([f"- {rtype}: {count} resources"
                             for rtype, count in resource_counts.items()])

        state["available_resources"] = list(resource_counts.keys())
        state["workflow_trace"].append(f"DISCOVER: Found {len(resource_counts)} resource types")

        return state

    def _identify_node(self, state: AgentState) -> AgentState:
        """Identify relevant resource types for the query."""
        state["workflow_trace"].append("IDENTIFY: Determining relevant resource types")

        prompt = self.IDENTIFY_PROMPT.format(
            question=state["question"],
            available_resources=", ".join(state["available_resources"]),
        )

        response = self.model.ask(prompt, max_new_tokens=500)

        # Parse response to extract relevant resources
        relevant = []
        for line in response.split("\n"):
            if "RELEVANT_RESOURCES:" in line.upper():
                resources_str = line.split(":", 1)[1].strip()
                relevant = [r.strip() for r in resources_str.split(",")]
                break

        # Filter to only available resources
        relevant = [r for r in relevant if r in state["available_resources"]]

        # Default to common types if none identified
        if not relevant:
            default_types = ["Condition", "Observation", "MedicationRequest", "Patient"]
            relevant = [r for r in default_types if r in state["available_resources"]]

        state["relevant_resource_types"] = relevant[:5]  # Limit to top 5
        state["workflow_trace"].append(f"IDENTIFY: Selected {len(relevant)} resource types")

        return state

    def _select_node(self, state: AgentState) -> AgentState:
        """Select specific resources to retrieve."""
        state["workflow_trace"].append("SELECT: Planning resource retrieval")

        bundle = state["fhir_bundle"]
        relevant_types = state["relevant_resource_types"]

        # Select resources of relevant types
        selected = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") in relevant_types:
                selected.append({
                    "resourceType": resource.get("resourceType"),
                    "id": resource.get("id"),
                    "resource": resource,
                })

        # Limit to prevent too many API calls
        state["selected_resources"] = selected[:20]
        state["workflow_trace"].append(f"SELECT: Identified {len(selected)} resources to examine")

        return state

    def _execute_node(self, state: AgentState) -> AgentState:
        """Execute retrieval and extract facts from resources."""
        state["workflow_trace"].append("EXECUTE: Extracting facts from resources")

        facts = []
        for resource_info in state["selected_resources"]:
            resource = resource_info["resource"]
            resource_type = resource_info["resourceType"]

            # Get human-readable summary of resource
            resource_summary = self._summarize_resource(resource)

            prompt = self.EXTRACT_PROMPT.format(
                question=state["question"],
                resource_type=resource_type,
                resource_data=resource_summary,
            )

            response = self.model.ask(prompt, max_new_tokens=600)

            # Parse facts from response
            if "NO_RELEVANT_FACTS" not in response.upper():
                extracted = self._parse_facts(
                    response,
                    resource_type,
                    resource_info["id"],
                )
                facts.extend(extracted)

        state["extracted_facts"] = [f.__dict__ for f in facts]
        state["workflow_trace"].append(f"EXECUTE: Extracted {len(facts)} relevant facts")

        return state

    def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize facts into a final answer."""
        state["workflow_trace"].append("SYNTHESIZE: Generating final answer")

        # Format facts for synthesis
        facts_text = "\n".join([
            f"- {f['content']} (from {f['source_resource_type']}, {f.get('date', 'date unknown')})"
            for f in state["extracted_facts"]
        ]) or "No specific facts were extracted from the records."

        prompt = self.SYNTHESIZE_PROMPT.format(
            question=state["question"],
            facts=facts_text,
            searched_resources=", ".join(state["relevant_resource_types"]),
        )

        response = self.model.ask(prompt, max_new_tokens=1000)

        # Extract answer
        answer = response
        for line in response.split("\n"):
            if "**ANSWER:**" in line or "ANSWER:" in line.upper():
                # Get content after this line
                idx = response.find(line)
                answer = response[idx:].split("**CONFIDENCE:**")[0].strip()
                answer = answer.replace("**ANSWER:**", "").replace("ANSWER:", "").strip()
                break

        state["answer"] = answer
        state["workflow_trace"].append("SYNTHESIZE: Answer generated")

        return state

    def _summarize_resource(self, resource: Dict[str, Any]) -> str:
        """Create a human-readable summary of a FHIR resource."""
        rtype = resource.get("resourceType", "Unknown")

        if rtype == "Patient":
            name = resource.get("name", [{}])[0]
            name_str = f"{name.get('given', [''])[0]} {name.get('family', '')}".strip()
            return f"Patient: {name_str}, DOB: {resource.get('birthDate', 'unknown')}, Gender: {resource.get('gender', 'unknown')}"

        elif rtype == "Condition":
            code = resource.get("code", {}).get("coding", [{}])[0]
            return f"Condition: {code.get('display', 'Unknown')}, Status: {resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', 'unknown')}, Onset: {resource.get('onsetDateTime', 'unknown')}"

        elif rtype == "Observation":
            code = resource.get("code", {}).get("coding", [{}])[0]
            value = resource.get("valueQuantity", {})
            value_str = f"{value.get('value', '')} {value.get('unit', '')}".strip() or resource.get("valueString", "")
            return f"Observation: {code.get('display', 'Unknown')} = {value_str}, Date: {resource.get('effectiveDateTime', 'unknown')}"

        elif rtype == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
            return f"Medication: {med.get('display', 'Unknown')}, Status: {resource.get('status', 'unknown')}, Authored: {resource.get('authoredOn', 'unknown')}"

        elif rtype == "Procedure":
            code = resource.get("code", {}).get("coding", [{}])[0]
            return f"Procedure: {code.get('display', 'Unknown')}, Date: {resource.get('performedDateTime', 'unknown')}, Status: {resource.get('status', 'unknown')}"

        elif rtype == "AllergyIntolerance":
            code = resource.get("code", {}).get("coding", [{}])[0]
            return f"Allergy: {code.get('display', 'Unknown')}, Criticality: {resource.get('criticality', 'unknown')}"

        else:
            # Generic summary
            return json.dumps(resource, indent=2)[:500]

    def _parse_facts(
        self,
        response: str,
        resource_type: str,
        resource_id: str,
    ) -> List[RetrievedFact]:
        """Parse extracted facts from model response."""
        facts = []
        current_fact = {}

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("FACT:"):
                if current_fact.get("content"):
                    facts.append(RetrievedFact(
                        content=current_fact["content"],
                        source_resource_type=resource_type,
                        source_resource_id=resource_id,
                        date=current_fact.get("date"),
                        relevance=current_fact.get("relevance", ""),
                    ))
                current_fact = {"content": line.split(":", 1)[1].strip()}

            elif line.upper().startswith("DATE:"):
                current_fact["date"] = line.split(":", 1)[1].strip()

            elif line.upper().startswith("RELEVANCE:"):
                current_fact["relevance"] = line.split(":", 1)[1].strip()

        # Don't forget the last fact
        if current_fact.get("content"):
            facts.append(RetrievedFact(
                content=current_fact["content"],
                source_resource_type=resource_type,
                source_resource_id=resource_id,
                date=current_fact.get("date"),
                relevance=current_fact.get("relevance", ""),
            ))

        return facts

    def query(
        self,
        question: str,
        fhir_bundle: Dict[str, Any],
    ) -> EHRQueryResult:
        """
        Query the EHR to answer a clinical question.

        Args:
            question: Natural language clinical question
            fhir_bundle: FHIR Bundle containing patient data

        Returns:
            EHRQueryResult with answer and supporting facts
        """
        if self._use_langgraph and self._workflow:
            # Use LangGraph workflow
            initial_state = {
                "question": question,
                "fhir_bundle": fhir_bundle,
                "available_resources": [],
                "relevant_resource_types": [],
                "selected_resources": [],
                "extracted_facts": [],
                "answer": "",
                "workflow_trace": [],
            }

            final_state = self._workflow.invoke(initial_state)

            # Convert extracted facts back to dataclass
            facts = [
                RetrievedFact(**f) for f in final_state["extracted_facts"]
            ]

            return EHRQueryResult(
                question=question,
                answer=final_state["answer"],
                facts=facts,
                resources_searched=final_state["relevant_resource_types"],
                workflow_trace=final_state["workflow_trace"],
            )

        else:
            # Fallback: simple direct query
            return self._simple_query(question, fhir_bundle)

    def _simple_query(
        self,
        question: str,
        fhir_bundle: Dict[str, Any],
    ) -> EHRQueryResult:
        """Simple fallback query without LangGraph."""
        # Summarize all resources
        summaries = []
        for entry in fhir_bundle.get("entry", [])[:20]:
            resource = entry.get("resource", {})
            summary = self._summarize_resource(resource)
            summaries.append(summary)

        prompt = f"""Answer this clinical question based on the patient's EHR data.

**Question:** {question}

**Patient Records:**
{chr(10).join(summaries)}

**ANSWER:**
Provide a comprehensive answer based on the available records.
"""

        answer = self.model.ask(prompt, max_new_tokens=1000)

        return EHRQueryResult(
            question=question,
            answer=answer,
            resources_searched=["All available"],
            workflow_trace=["SIMPLE_QUERY: Direct analysis"],
        )

    def load_fhir_bundle(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a FHIR bundle from a JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def get_patient_summary(self, fhir_bundle: Dict[str, Any]) -> str:
        """Generate a quick summary of patient from FHIR bundle."""
        summaries = {
            "patient": [],
            "conditions": [],
            "medications": [],
            "allergies": [],
        }

        for entry in fhir_bundle.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")

            if rtype == "Patient":
                summaries["patient"].append(self._summarize_resource(resource))
            elif rtype == "Condition":
                summaries["conditions"].append(self._summarize_resource(resource))
            elif rtype == "MedicationRequest":
                summaries["medications"].append(self._summarize_resource(resource))
            elif rtype == "AllergyIntolerance":
                summaries["allergies"].append(self._summarize_resource(resource))

        lines = ["**PATIENT SUMMARY**", ""]

        if summaries["patient"]:
            lines.append("Demographics:")
            lines.extend([f"  {s}" for s in summaries["patient"]])

        if summaries["conditions"]:
            lines.append("\nActive Conditions:")
            lines.extend([f"  - {s}" for s in summaries["conditions"][:10]])

        if summaries["medications"]:
            lines.append("\nMedications:")
            lines.extend([f"  - {s}" for s in summaries["medications"][:10]])

        if summaries["allergies"]:
            lines.append("\nAllergies:")
            lines.extend([f"  - {s}" for s in summaries["allergies"]])

        return "\n".join(lines)
