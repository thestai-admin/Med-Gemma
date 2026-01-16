"""
Guidelines Agent - Evidence-Based Clinical Practice Guidelines RAG

Retrieves and synthesizes relevant clinical practice guidelines
based on the patient's differential diagnosis and chief complaint.

Uses RAG (Retrieval-Augmented Generation) with:
- Pre-computed embeddings from sentence-transformers
- Cosine similarity for retrieval
- MedGemma for synthesis

This agent provides evidence-based recommendations with citations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import numpy as np


@dataclass
class GuidelineChunk:
    """A chunk of clinical guideline text with metadata."""
    guideline_name: str      # e.g., "ACC/AHA Chest Pain Guidelines 2021"
    section: str             # e.g., "Initial Evaluation"
    content: str             # Text chunk (~300 tokens)
    source: str              # Citation/reference
    evidence_level: str      # "Level A", "Level B", "Level C", "Expert Consensus"
    condition: str = ""      # Primary condition covered

    def to_citation(self) -> str:
        """Format as citation."""
        return f"{self.guideline_name} - {self.section} ({self.evidence_level})"


@dataclass
class GuidelineRecommendation:
    """A synthesized guideline recommendation."""
    recommendation: str
    evidence_level: str
    source_guidelines: List[str] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass
class GuidelinesResult:
    """Complete result from guidelines retrieval and synthesis."""
    recommendations: List[GuidelineRecommendation] = field(default_factory=list)
    retrieved_chunks: List[GuidelineChunk] = field(default_factory=list)
    summary: str = ""
    conditions_matched: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format for inclusion in reports."""
        if not self.recommendations:
            return "No specific guideline recommendations available."

        lines = []
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"{i}. {rec.recommendation}")
            if rec.source_guidelines:
                sources = ", ".join(rec.source_guidelines[:2])
                lines.append(f"   Source: {sources} [{rec.evidence_level}]")

        return "\n".join(lines)

    def to_report_section(self) -> str:
        """Generate formatted section for clinical report."""
        if not self.recommendations:
            return ""

        lines = [
            "-" * 40,
            "EVIDENCE-BASED GUIDELINES",
            "-" * 40,
        ]

        if self.conditions_matched:
            lines.append(f"Guidelines consulted for: {', '.join(self.conditions_matched)}")
            lines.append("")

        lines.append("**Recommendations:**")
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"{i}. {rec.recommendation}")
            lines.append(f"   Evidence: {rec.evidence_level}")
            if rec.source_guidelines:
                lines.append(f"   Source: {rec.source_guidelines[0]}")
            lines.append("")

        if self.summary:
            lines.append("**Summary:**")
            lines.append(self.summary)
            lines.append("")

        # Add citations
        if self.retrieved_chunks:
            lines.append("**References:**")
            seen_sources = set()
            for chunk in self.retrieved_chunks[:5]:
                citation = chunk.to_citation()
                if citation not in seen_sources:
                    lines.append(f"- {citation}")
                    seen_sources.add(citation)

        return "\n".join(lines)


class GuidelinesAgent:
    """
    Agent for retrieving and synthesizing clinical practice guidelines.

    Uses RAG (Retrieval-Augmented Generation) to provide evidence-based
    recommendations based on the patient's clinical presentation.

    Usage:
        agent = GuidelinesAgent()
        result = agent.get_recommendations(
            differential_diagnosis=["Pneumonia", "COPD Exacerbation"],
            chief_complaint="Cough with fever"
        )
    """

    # Synthesis prompt for MedGemma
    SYNTHESIS_PROMPT = """You are a clinical guidelines expert. Based on the following retrieved guideline excerpts, provide evidence-based recommendations for the patient.

## Patient Presentation
Chief Complaint: {chief_complaint}
Differential Diagnosis: {differential}

## Retrieved Guidelines
{guidelines_context}

---

Based on these guidelines, provide 3-5 specific, actionable recommendations. For each recommendation:
1. State the recommendation clearly
2. Note the evidence level
3. Cite the source guideline

Format your response as:

**RECOMMENDATIONS:**
1. [Recommendation text]
   Evidence: [Level A/B/C/Expert Consensus]
   Source: [Guideline name]

2. [Next recommendation...]

**SUMMARY:**
[2-3 sentence summary of key guideline-based considerations for this patient]"""

    def __init__(
        self,
        model=None,
        embeddings_path: Optional[str] = None,
        chunks_path: Optional[str] = None,
    ):
        """
        Initialize the Guidelines Agent.

        Args:
            model: Optional MedGemma model instance for synthesis
            embeddings_path: Path to pre-computed embeddings (.npz file)
            chunks_path: Path to guideline chunks JSON file
        """
        self._model = model
        self._embedder = None
        self._embeddings = None
        self._chunks: List[GuidelineChunk] = []

        # Default paths
        base_path = Path(__file__).parent.parent.parent / "data" / "guidelines"
        self._embeddings_path = Path(embeddings_path) if embeddings_path else base_path / "embeddings.npz"
        self._chunks_path = Path(chunks_path) if chunks_path else base_path / "chunks.json"

        # Track if loaded
        self._loaded = False

    @property
    def model(self):
        """Lazy load MedGemma model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def _load_embedder(self):
        """Load sentence-transformers model for embedding queries."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Small, fast model that runs on CPU
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded sentence-transformers embedder")
            except ImportError:
                print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
        return self._embedder

    def _load_data(self):
        """Load pre-computed embeddings and guideline chunks."""
        if self._loaded:
            return

        # Load chunks
        if self._chunks_path.exists():
            with open(self._chunks_path, 'r') as f:
                chunks_data = json.load(f)
            self._chunks = [
                GuidelineChunk(**chunk) for chunk in chunks_data
            ]
            print(f"Loaded {len(self._chunks)} guideline chunks")
        else:
            print(f"Warning: Chunks file not found at {self._chunks_path}")
            self._chunks = []

        # Load embeddings
        if self._embeddings_path.exists():
            data = np.load(self._embeddings_path)
            self._embeddings = data['embeddings']
            print(f"Loaded embeddings with shape {self._embeddings.shape}")
        else:
            print(f"Warning: Embeddings file not found at {self._embeddings_path}")
            self._embeddings = None

        self._loaded = True

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        embedder = self._load_embedder()
        return embedder.encode(query, convert_to_numpy=True)

    def _retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[tuple]:
        """
        Retrieve relevant guideline chunks for a query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of (GuidelineChunk, similarity_score) tuples
        """
        self._load_data()

        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Embed query
        query_embedding = self._embed_query(query)

        # Compute cosine similarity
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self._embeddings / np.linalg.norm(self._embeddings, axis=1, keepdims=True)

        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                results.append((self._chunks[idx], float(score)))

        return results

    def get_recommendations(
        self,
        differential_diagnosis: List[str],
        chief_complaint: str,
        top_k: int = 5,
    ) -> GuidelinesResult:
        """
        Get guideline-based recommendations for a clinical presentation.

        Args:
            differential_diagnosis: List of diagnoses to look up
            chief_complaint: Patient's chief complaint
            top_k: Number of guideline chunks to retrieve

        Returns:
            GuidelinesResult with recommendations and sources
        """
        result = GuidelinesResult()

        # Build comprehensive query
        query_parts = [chief_complaint]
        query_parts.extend(differential_diagnosis[:3])  # Top 3 diagnoses
        query = " ".join(query_parts)

        # Retrieve relevant chunks
        retrieved = self._retrieve(query, top_k=top_k)

        if not retrieved:
            result.summary = "No matching clinical guidelines found in the knowledge base."
            return result

        # Store retrieved chunks
        result.retrieved_chunks = [chunk for chunk, _ in retrieved]
        result.conditions_matched = list(set(
            chunk.condition for chunk in result.retrieved_chunks if chunk.condition
        ))

        # Build context for synthesis
        guidelines_context = self._format_retrieved_chunks(retrieved)

        # Synthesize recommendations with MedGemma
        prompt = self.SYNTHESIS_PROMPT.format(
            chief_complaint=chief_complaint,
            differential=", ".join(differential_diagnosis),
            guidelines_context=guidelines_context,
        )

        try:
            response = self.model.ask(prompt, max_new_tokens=1500)
            result.recommendations = self._parse_recommendations(response)
            result.summary = self._extract_summary(response)
        except Exception as e:
            print(f"Error synthesizing guidelines: {e}")
            # Fall back to returning raw chunks
            result.summary = f"Retrieved {len(retrieved)} relevant guideline sections."

        return result

    def _format_retrieved_chunks(self, retrieved: List[tuple]) -> str:
        """Format retrieved chunks for the synthesis prompt."""
        lines = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            lines.append(f"### Guideline {i}: {chunk.guideline_name}")
            lines.append(f"Section: {chunk.section}")
            lines.append(f"Evidence Level: {chunk.evidence_level}")
            lines.append(f"Content: {chunk.content}")
            lines.append(f"Relevance: {score:.2f}")
            lines.append("")
        return "\n".join(lines)

    def _parse_recommendations(self, response: str) -> List[GuidelineRecommendation]:
        """Parse recommendations from model response."""
        recommendations = []

        # Find recommendations section
        if "**RECOMMENDATIONS:**" in response:
            rec_section = response.split("**RECOMMENDATIONS:**")[1]
            if "**SUMMARY:**" in rec_section:
                rec_section = rec_section.split("**SUMMARY:**")[0]
        else:
            rec_section = response

        # Parse numbered recommendations
        current_rec = None
        for line in rec_section.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for numbered recommendation
            if line[0].isdigit() and "." in line:
                if current_rec:
                    recommendations.append(current_rec)

                content = line.split(".", 1)[1].strip()
                current_rec = GuidelineRecommendation(
                    recommendation=content,
                    evidence_level="",
                    source_guidelines=[],
                )
            elif current_rec:
                # Parse evidence level
                lower = line.lower()
                if "evidence:" in lower:
                    level = line.split(":", 1)[1].strip()
                    current_rec.evidence_level = level
                elif "source:" in lower:
                    source = line.split(":", 1)[1].strip()
                    current_rec.source_guidelines.append(source)

        if current_rec:
            recommendations.append(current_rec)

        return recommendations

    def _extract_summary(self, response: str) -> str:
        """Extract summary from model response."""
        if "**SUMMARY:**" in response:
            summary = response.split("**SUMMARY:**")[1].strip()
            # Take first paragraph
            summary = summary.split("\n\n")[0].strip()
            return summary
        return ""

    def search_guidelines(
        self,
        condition: str,
        top_k: int = 3,
    ) -> List[GuidelineChunk]:
        """
        Search for guidelines related to a specific condition.

        Args:
            condition: Condition to search for
            top_k: Number of results

        Returns:
            List of relevant GuidelineChunk objects
        """
        retrieved = self._retrieve(condition, top_k=top_k)
        return [chunk for chunk, _ in retrieved]
