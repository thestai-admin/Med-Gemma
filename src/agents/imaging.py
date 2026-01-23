"""
Imaging Agent - Medical Image Analysis

Analyzes medical images (primarily chest X-rays) using MedGemma
to extract findings, generate reports, and classify pathologies.

Combines:
- MedGemma for detailed image analysis and report generation
- MedSigLIP for zero-shot classification
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from PIL import Image
from enum import Enum


class ImageModality(Enum):
    """Supported imaging modalities."""
    CHEST_XRAY = "chest_xray"
    CT = "ct"
    MRI = "mri"
    UNKNOWN = "unknown"


class FindingSeverity(Enum):
    """Severity of imaging findings."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ChangeStatus(Enum):
    """Change status for longitudinal comparison."""
    NEW = "new"
    IMPROVED = "improved"
    WORSENED = "worsened"
    UNCHANGED = "unchanged"
    RESOLVED = "resolved"


@dataclass
class ImagingFinding:
    """Individual finding from image analysis."""
    description: str
    location: Optional[str] = None
    severity: FindingSeverity = FindingSeverity.MILD
    confidence: Optional[float] = None


@dataclass
class ImageAnalysis:
    """Complete analysis of a medical image."""
    modality: ImageModality
    technique: str
    quality: str
    findings: List[ImagingFinding] = field(default_factory=list)
    impression: str = ""
    classification_probs: Dict[str, float] = field(default_factory=dict)
    raw_analysis: str = ""
    requires_urgent_review: bool = False

    def to_prompt_context(self) -> str:
        """Format analysis for downstream prompts."""
        lines = [
            f"**Imaging Modality:** {self.modality.value}",
            f"**Technical Quality:** {self.quality}",
        ]

        if self.findings:
            lines.append("\n**Findings:**")
            for i, finding in enumerate(self.findings, 1):
                loc = f" ({finding.location})" if finding.location else ""
                lines.append(f"{i}. {finding.description}{loc}")

        if self.impression:
            lines.append(f"\n**Impression:** {self.impression}")

        if self.classification_probs:
            lines.append("\n**Classification Probabilities:**")
            sorted_probs = sorted(
                self.classification_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for label, prob in sorted_probs[:5]:
                lines.append(f"- {label}: {prob*100:.1f}%")

        return "\n".join(lines)

    def get_top_classifications(self, n: int = 3) -> List[tuple]:
        """Get top N classifications by probability."""
        sorted_probs = sorted(
            self.classification_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_probs[:n]


@dataclass
class LongitudinalAnalysis:
    """Analysis comparing two images over time (prior vs current)."""
    prior_findings: str
    current_findings: str
    comparison: str
    change_summary: ChangeStatus
    interval: Optional[str] = None  # Time between studies if known
    key_changes: List[str] = field(default_factory=list)
    requires_urgent_review: bool = False
    raw_prior_analysis: str = ""
    raw_current_analysis: str = ""

    def to_prompt_context(self) -> str:
        """Format longitudinal analysis for downstream prompts."""
        lines = [
            "**LONGITUDINAL IMAGING COMPARISON**",
            "",
        ]

        if self.interval:
            lines.append(f"**Interval:** {self.interval}")

        lines.extend([
            "",
            "**Prior Study Findings:**",
            self.prior_findings,
            "",
            "**Current Study Findings:**",
            self.current_findings,
            "",
            "**Comparison Summary:**",
            self.comparison,
            "",
            f"**Overall Change:** {self.change_summary.value.upper()}",
        ])

        if self.key_changes:
            lines.append("")
            lines.append("**Key Changes:**")
            for change in self.key_changes:
                lines.append(f"- {change}")

        if self.requires_urgent_review:
            lines.append("")
            lines.append("**⚠️ URGENT: Significant changes requiring immediate attention**")

        return "\n".join(lines)

    def to_report_section(self) -> str:
        """Generate a report section for longitudinal comparison."""
        return f"""
----------------------------------------
COMPARISON WITH PRIOR STUDY
----------------------------------------
{self.comparison}

Change Assessment: {self.change_summary.value.upper()}
{"URGENT REVIEW RECOMMENDED" if self.requires_urgent_review else ""}
"""


class ImagingAgent:
    """
    Agent for medical image analysis.

    Uses MedGemma for detailed analysis and MedSigLIP for classification.

    Usage:
        agent = ImagingAgent()
        analysis = agent.analyze(image, clinical_context="65yo smoker with cough")
    """

    # Enhanced classification labels with descriptive prompts for better accuracy
    CXR_LABELS = [
        "chest x-ray showing normal lung fields with clear costophrenic angles",
        "chest x-ray showing pneumonia with consolidation or infiltrates",
        "chest x-ray showing pleural effusion with blunted costophrenic angle",
        "chest x-ray showing cardiomegaly with enlarged cardiac silhouette",
        "chest x-ray showing pulmonary edema with bilateral infiltrates",
        "chest x-ray showing atelectasis with volume loss",
        "chest x-ray showing pneumothorax with absent lung markings",
        "chest x-ray showing consolidation with air bronchograms",
        "chest x-ray showing mass or nodule in lung field",
        "chest x-ray showing interstitial lung disease with reticular pattern",
    ]

    # Simple labels for backward compatibility and mapping
    CXR_LABELS_SIMPLE = [
        "normal chest x-ray",
        "pneumonia",
        "pleural effusion",
        "cardiomegaly",
        "pulmonary edema",
        "atelectasis",
        "pneumothorax",
        "consolidation",
        "mass or nodule",
        "interstitial lung disease",
    ]

    # Binary labels for pneumonia detection (high accuracy mode)
    PNEUMONIA_BINARY_LABELS = [
        "normal healthy chest x-ray with clear lungs and no infiltrates",
        "chest x-ray showing pneumonia, infection, consolidation, or infiltrates",
    ]

    # Prompt for systematic chest X-ray analysis
    CXR_ANALYSIS_PROMPT = """Analyze this chest X-ray systematically.

{context}

Provide a structured analysis:

**TECHNIQUE:**
Describe the imaging technique (PA/AP/lateral, inspiration quality, rotation, exposure)

**QUALITY:**
Rate image quality (adequate/suboptimal) and note any limitations

**FINDINGS:**
Systematically describe:
1. Cardiac silhouette (size, contour, position)
2. Mediastinum (width, contour, tracheal position)
3. Lungs (opacity, lucency, distribution, lung volumes)
4. Pleura (effusions, thickening, pneumothorax)
5. Bones and soft tissues (fractures, lesions, soft tissue abnormalities)
6. Lines and tubes (if present)

**IMPRESSION:**
Concise summary of key findings

**URGENT:**
[YES/NO] - Are there findings requiring immediate attention?"""

    # Prompt for findings-focused analysis
    FINDINGS_PROMPT = """List all abnormal findings in this chest X-ray.

{context}

For each finding, provide:
1. Description of the abnormality
2. Location (e.g., right lower lobe, left hilum)
3. Severity (normal/mild/moderate/severe)

If the image appears normal, state "No acute abnormality identified."
"""

    # Prompt for longitudinal comparison
    LONGITUDINAL_PRIOR_PROMPT = """Analyze this PRIOR chest X-ray study.

{context}

Provide a concise summary of findings that can be compared to a follow-up study:
1. Key findings (abnormalities, their location, and severity)
2. Overall lung volumes and cardiac size
3. Any lines, tubes, or devices present
4. Technical quality of the study
"""

    LONGITUDINAL_CURRENT_PROMPT = """Analyze this CURRENT chest X-ray study.

{context}

Provide a concise summary of findings that can be compared to a prior study:
1. Key findings (abnormalities, their location, and severity)
2. Overall lung volumes and cardiac size
3. Any lines, tubes, or devices present
4. Technical quality of the study
"""

    LONGITUDINAL_COMPARISON_PROMPT = """Compare these two chest X-ray studies over time.

**PRIOR STUDY FINDINGS:**
{prior_findings}

**CURRENT STUDY FINDINGS:**
{current_findings}

{context}

Provide a detailed comparison:

**COMPARISON:**
Describe interval changes in:
1. Any new findings not present on the prior study
2. Findings that have resolved since the prior study
3. Findings that have improved (decreased size, density, or extent)
4. Findings that have worsened (increased size, density, or extent)
5. Findings that remain unchanged

**KEY CHANGES:**
List the most clinically significant changes (bullet points)

**CHANGE SUMMARY:**
Provide ONE word overall assessment: NEW, IMPROVED, WORSENED, UNCHANGED, or RESOLVED

**URGENT:**
[YES/NO] - Are there findings requiring immediate attention?
"""

    def __init__(self, model=None, classifier=None, load_classifier: bool = True):
        """
        Initialize the Imaging Agent.

        Args:
            model: Optional MedGemma model instance
            classifier: Optional MedSigLIP classifier instance
            load_classifier: Whether to load classifier (default True)
        """
        self._model = model
        self._classifier = classifier
        self._load_classifier = load_classifier

    @property
    def model(self):
        """Lazy load MedGemma model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    @property
    def classifier(self):
        """Lazy load MedSigLIP classifier."""
        if self._classifier is None and self._load_classifier:
            from ..model import MedSigLIP
            self._classifier = MedSigLIP()
        return self._classifier

    def analyze(
        self,
        image: Union[Image.Image, str, Path],
        clinical_context: Optional[str] = None,
        include_classification: bool = True,
        modality: ImageModality = ImageModality.CHEST_XRAY,
        skip_classification_if_confident: bool = True,
    ) -> ImageAnalysis:
        """
        Perform complete image analysis.

        Args:
            image: PIL Image or path to image
            clinical_context: Optional clinical information
            include_classification: Whether to run zero-shot classification
            modality: Type of imaging
            skip_classification_if_confident: Skip MedSigLIP if MedGemma is confident (saves ~3-5s)

        Returns:
            ImageAnalysis with findings, impression, and classifications
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Build context for prompt
        context = ""
        if clinical_context:
            context = f"Clinical Context: {clinical_context}\n"

        # Run primary analysis with MedGemma (reduced tokens for latency)
        if modality == ImageModality.CHEST_XRAY:
            prompt = self.CXR_ANALYSIS_PROMPT.format(context=context)
        else:
            prompt = f"Analyze this medical image in detail.\n{context}"

        raw_analysis = self.model.analyze_image(image, prompt, max_new_tokens=1500)

        # Parse the analysis
        analysis = self._parse_analysis(raw_analysis, modality)

        # Run classification if requested
        if include_classification and self.classifier:
            # Check if MedGemma analysis is confident (skip classification for latency)
            skip_classification = False
            if skip_classification_if_confident:
                raw_lower = raw_analysis.lower()
                # Skip if MedGemma is confident (no uncertainty markers)
                uncertainty_markers = ["uncertain", "unclear", "cannot determine",
                                       "difficult to assess", "limited", "suboptimal"]
                has_uncertainty = any(marker in raw_lower for marker in uncertainty_markers)
                skip_classification = not has_uncertainty

            if not skip_classification:
                labels = self.CXR_LABELS if modality == ImageModality.CHEST_XRAY else []
                if labels:
                    probs = self.classifier.classify(image, labels)
                    # Map to simple labels
                    analysis.classification_probs = {
                        simple: probs[enhanced]
                        for simple, enhanced in zip(self.CXR_LABELS_SIMPLE, self.CXR_LABELS)
                    }

        return analysis

    def get_findings(
        self,
        image: Union[Image.Image, str, Path],
        clinical_context: Optional[str] = None,
    ) -> List[ImagingFinding]:
        """
        Extract just the findings from an image.

        Args:
            image: PIL Image or path
            clinical_context: Optional clinical information

        Returns:
            List of ImagingFinding objects
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        context = f"Clinical Context: {clinical_context}" if clinical_context else ""
        prompt = self.FINDINGS_PROMPT.format(context=context)

        response = self.model.analyze_image(image, prompt, max_new_tokens=1500)
        return self._parse_findings(response)

    def classify(
        self,
        image: Union[Image.Image, str, Path],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Zero-shot classification of image.

        Args:
            image: PIL Image or path
            labels: Optional custom labels (defaults to CXR_LABELS)

        Returns:
            Dictionary of label -> probability
        """
        if not self.classifier:
            raise RuntimeError("Classifier not loaded")

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        labels = labels or self.CXR_LABELS
        probs = self.classifier.classify(image, labels)

        # Map enhanced labels back to simple labels for compatibility
        if labels == self.CXR_LABELS:
            return {
                simple: probs[enhanced]
                for simple, enhanced in zip(self.CXR_LABELS_SIMPLE, self.CXR_LABELS)
            }
        return probs

    def classify_pneumonia_binary(
        self,
        image: Union[Image.Image, str, Path],
    ) -> Dict[str, float]:
        """
        Binary pneumonia vs normal classification with optimized prompts.

        More accurate than multi-label classification for pneumonia detection.

        Args:
            image: PIL Image or path

        Returns:
            Dictionary with 'normal' and 'pneumonia' probabilities
        """
        if not self.classifier:
            raise RuntimeError("Classifier not loaded")

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        probs = self.classifier.classify(image, self.PNEUMONIA_BINARY_LABELS)

        # Return with simple keys
        return {
            "normal": probs[self.PNEUMONIA_BINARY_LABELS[0]],
            "pneumonia": probs[self.PNEUMONIA_BINARY_LABELS[1]],
        }

    def classify_with_ensemble(
        self,
        image: Union[Image.Image, str, Path],
        clinical_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ensemble classification combining MedGemma analysis with MedSigLIP.

        Improves accuracy by combining:
        - MedGemma's textual analysis for abnormality detection
        - MedSigLIP's binary pneumonia classification

        Args:
            image: PIL Image or path
            clinical_context: Optional clinical information

        Returns:
            Dictionary with ensemble results:
            - pneumonia_score: Combined probability (0-1)
            - siglip_probs: Raw MedSigLIP binary probabilities
            - gemma_abnormal: Whether MedGemma detected abnormality
            - is_pneumonia: Final binary decision
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # 1. Get MedSigLIP binary classification
        siglip_probs = self.classify_pneumonia_binary(image)

        # 2. Get MedGemma quick abnormality assessment
        quick_prompt = """Look at this chest x-ray and answer with a single word:
Is this chest x-ray NORMAL or ABNORMAL?
Answer:"""
        gemma_response = self.model.analyze_image(image, quick_prompt, max_new_tokens=20)
        gemma_abnormal = "ABNORMAL" in gemma_response.upper()

        # 3. Combine signals with weighted ensemble
        # MedSigLIP weight: 0.4, MedGemma weight: 0.6 (MedGemma is more reliable)
        siglip_score = siglip_probs["pneumonia"]
        gemma_score = 0.7 if gemma_abnormal else 0.1  # High confidence if abnormal

        ensemble_score = siglip_score * 0.4 + gemma_score * 0.6

        # Threshold for pneumonia detection
        is_pneumonia = ensemble_score > 0.4

        return {
            "pneumonia_score": ensemble_score,
            "siglip_probs": siglip_probs,
            "gemma_abnormal": gemma_abnormal,
            "gemma_response": gemma_response.strip(),
            "is_pneumonia": is_pneumonia,
        }

    def generate_report(
        self,
        image: Union[Image.Image, str, Path],
        patient_info: Optional[Dict[str, Any]] = None,
        comparison: Optional[str] = None,
    ) -> str:
        """
        Generate a formal radiology-style report.

        Args:
            image: PIL Image or path
            patient_info: Optional dict with age, gender, history
            comparison: Note about prior studies for comparison

        Returns:
            Formatted radiology report string
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Build patient context
        context_parts = []
        if patient_info:
            if patient_info.get("age"):
                context_parts.append(f"Age: {patient_info['age']}")
            if patient_info.get("gender"):
                context_parts.append(f"Gender: {patient_info['gender']}")
            if patient_info.get("history"):
                context_parts.append(f"Clinical History: {patient_info['history']}")

        context = "\n".join(context_parts) if context_parts else "No clinical context provided"

        prompt = f"""Generate a complete radiology report for this chest X-ray.

Patient Information:
{context}

{f'Comparison: {comparison}' if comparison else 'Comparison: None available'}

Format as a formal radiology report with:

**EXAMINATION:** Chest X-ray

**CLINICAL INDICATION:** [from context]

**TECHNIQUE:** [describe technique]

**COMPARISON:** [prior studies]

**FINDINGS:**
[Detailed systematic findings]

**IMPRESSION:**
[Numbered list of key findings/diagnoses]

**RECOMMENDATIONS:**
[If applicable]"""

        return self.model.analyze_image(image, prompt, max_new_tokens=2500)

    def _parse_analysis(self, raw: str, modality: ImageModality) -> ImageAnalysis:
        """Parse raw analysis text into structured ImageAnalysis."""
        analysis = ImageAnalysis(
            modality=modality,
            technique="",
            quality="",
            raw_analysis=raw,
        )

        # Extract sections
        sections = self._extract_sections(raw)

        analysis.technique = sections.get("TECHNIQUE", "Standard")
        analysis.quality = sections.get("QUALITY", "Adequate")
        analysis.impression = sections.get("IMPRESSION", "")

        # Check for urgency
        urgent = sections.get("URGENT", "").upper()
        analysis.requires_urgent_review = "YES" in urgent

        # Parse findings
        findings_text = sections.get("FINDINGS", "")
        if findings_text:
            analysis.findings = self._parse_findings(findings_text)

        return analysis

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract labeled sections from response."""
        sections = {}
        current_section = None
        current_content = []

        for line in text.split("\n"):
            line = line.strip()

            # Check for section header
            if line.startswith("**") and line.endswith("**"):
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                # Start new section
                current_section = line.strip("*: ").upper()
                current_content = []
            elif ":" in line and line.split(":")[0].strip("*").upper() in [
                "TECHNIQUE", "QUALITY", "FINDINGS", "IMPRESSION", "URGENT"
            ]:
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                parts = line.split(":", 1)
                current_section = parts[0].strip("* ").upper()
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            elif current_section:
                current_content.append(line)

        # Save final section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _parse_findings(self, text: str) -> List[ImagingFinding]:
        """Parse findings text into list of ImagingFinding."""
        findings = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Skip headers and labels
            if line.startswith("**") or line.endswith(":"):
                continue

            # Parse numbered or bulleted items
            if line[0].isdigit() or line.startswith("-") or line.startswith("•"):
                # Remove bullet/number
                content = line.lstrip("0123456789.-•) ").strip()
                if content and len(content) > 5:
                    # Try to extract location
                    location = None
                    for loc in ["right", "left", "bilateral", "upper", "lower", "middle"]:
                        if loc in content.lower():
                            location = loc
                            break

                    findings.append(ImagingFinding(
                        description=content,
                        location=location,
                    ))

        return findings

    def analyze_longitudinal(
        self,
        prior_image: Union[Image.Image, str, Path],
        current_image: Union[Image.Image, str, Path],
        clinical_context: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> LongitudinalAnalysis:
        """
        Compare two chest X-rays over time to assess disease progression.

        IMPORTANT: Processes images sequentially to avoid GPU OOM on T4.

        Args:
            prior_image: Prior/baseline chest X-ray
            current_image: Current/follow-up chest X-ray
            clinical_context: Optional clinical information
            interval: Time between studies (e.g., "6 months", "2 weeks")

        Returns:
            LongitudinalAnalysis with comparison results
        """
        import torch

        # Load images if paths provided
        if isinstance(prior_image, (str, Path)):
            prior_image = Image.open(prior_image)
        if isinstance(current_image, (str, Path)):
            current_image = Image.open(current_image)

        # Build context for prompts
        context = ""
        if clinical_context:
            context = f"Clinical Context: {clinical_context}"
        if interval:
            context += f"\nInterval since prior study: {interval}"

        # Step 1: Analyze PRIOR study
        print("Analyzing prior study...")
        prior_prompt = self.LONGITUDINAL_PRIOR_PROMPT.format(context=context)
        prior_findings = self.model.analyze_image(prior_image, prior_prompt, max_new_tokens=1000)

        # Clear GPU cache between analyses (critical for T4 16GB)
        torch.cuda.empty_cache()

        # Step 2: Analyze CURRENT study
        print("Analyzing current study...")
        current_prompt = self.LONGITUDINAL_CURRENT_PROMPT.format(context=context)
        current_findings = self.model.analyze_image(current_image, current_prompt, max_new_tokens=1000)

        # Clear GPU cache before synthesis
        torch.cuda.empty_cache()

        # Step 3: Generate comparison (text-only, no images needed)
        print("Generating comparison...")
        comparison_prompt = self.LONGITUDINAL_COMPARISON_PROMPT.format(
            prior_findings=prior_findings,
            current_findings=current_findings,
            context=context,
        )
        comparison_response = self.model.ask(comparison_prompt, max_new_tokens=1500)

        # Parse the comparison response
        return self._parse_longitudinal_comparison(
            prior_findings=prior_findings,
            current_findings=current_findings,
            comparison_response=comparison_response,
            interval=interval,
        )

    def _parse_longitudinal_comparison(
        self,
        prior_findings: str,
        current_findings: str,
        comparison_response: str,
        interval: Optional[str] = None,
    ) -> LongitudinalAnalysis:
        """Parse longitudinal comparison response into structured result."""
        sections = self._extract_sections(comparison_response)

        # Extract comparison text
        comparison = sections.get("COMPARISON", comparison_response)

        # Extract key changes
        key_changes = []
        key_changes_text = sections.get("KEY CHANGES", "")
        for line in key_changes_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                content = line.lstrip("0123456789.-•) ").strip()
                if content:
                    key_changes.append(content)

        # Extract change summary
        change_summary_text = sections.get("CHANGE SUMMARY", "").upper().strip()
        change_summary = ChangeStatus.UNCHANGED  # Default
        if "NEW" in change_summary_text:
            change_summary = ChangeStatus.NEW
        elif "IMPROVED" in change_summary_text:
            change_summary = ChangeStatus.IMPROVED
        elif "WORSENED" in change_summary_text or "WORSE" in change_summary_text:
            change_summary = ChangeStatus.WORSENED
        elif "RESOLVED" in change_summary_text:
            change_summary = ChangeStatus.RESOLVED

        # Check urgency
        urgent_text = sections.get("URGENT", "").upper()
        requires_urgent_review = "YES" in urgent_text

        return LongitudinalAnalysis(
            prior_findings=prior_findings,
            current_findings=current_findings,
            comparison=comparison,
            change_summary=change_summary,
            interval=interval,
            key_changes=key_changes,
            requires_urgent_review=requires_urgent_review,
            raw_prior_analysis=prior_findings,
            raw_current_analysis=current_findings,
        )
