"""
Inference Pipeline for Medical AI

Combines MedGemma with data loading for end-to-end medical image analysis.
Designed for the MedGemma Impact Challenge.

Features:
- Chest X-ray analysis and report generation
- Differential diagnosis generation
- Batch processing support
- Result formatting for clinical use
"""

import torch
from PIL import Image
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .model import MedGemma, MedSigLIP, ModelIDs, PROMPTS
from .data import NIHChestXray, get_sample_chest_xrays


# =============================================================================
# Analysis Types
# =============================================================================

class AnalysisType(Enum):
    """Types of medical image analysis."""
    DESCRIBE = "describe"
    FINDINGS = "findings"
    DIFFERENTIAL = "differential"
    REPORT = "report"
    PRIMARY_CARE = "primary_care"


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of a medical image analysis."""
    image_id: str
    analysis_type: str
    response: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChestXrayReport:
    """Structured chest X-ray report."""
    findings: str
    impression: str
    differential: List[str]
    recommendations: List[str]
    raw_response: str


# =============================================================================
# Medical Image Analyzer
# =============================================================================

class MedicalImageAnalyzer:
    """
    High-level interface for medical image analysis.

    Combines MedGemma for generation and MedSigLIP for classification.

    Usage:
        analyzer = MedicalImageAnalyzer()
        result = analyzer.analyze(image, AnalysisType.REPORT)
        print(result.response)
    """

    def __init__(
        self,
        model_id: str = ModelIDs.MEDGEMMA_1_5_4B,
        device: str = "cuda",
        load_classifier: bool = False,
    ):
        """
        Initialize the analyzer.

        Args:
            model_id: MedGemma model ID
            device: Device to run on
            load_classifier: Also load MedSigLIP for classification
        """
        self.device = device
        self.model = MedGemma(model_id=model_id, device=device)

        if load_classifier:
            self.classifier = MedSigLIP(device=device)
        else:
            self.classifier = None

    def analyze(
        self,
        image: Union[Image.Image, str, Path],
        analysis_type: AnalysisType = AnalysisType.DESCRIBE,
        custom_prompt: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> AnalysisResult:
        """
        Analyze a medical image.

        Args:
            image: PIL Image or path to image
            analysis_type: Type of analysis to perform
            custom_prompt: Override the default prompt
            max_tokens: Maximum tokens to generate

        Returns:
            AnalysisResult with the response
        """
        if isinstance(image, (str, Path)):
            image_id = str(image)
            image = Image.open(image)
        else:
            image_id = "uploaded_image"

        # Get appropriate prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = PROMPTS.get(analysis_type.value, PROMPTS["describe"])

        # Run analysis
        response = self.model.analyze_image(image, prompt, max_new_tokens=max_tokens)

        return AnalysisResult(
            image_id=image_id,
            analysis_type=analysis_type.value,
            response=response,
        )

    def generate_report(
        self,
        image: Union[Image.Image, str, Path],
        patient_info: Optional[Dict[str, str]] = None,
    ) -> ChestXrayReport:
        """
        Generate a structured chest X-ray report.

        Args:
            image: Chest X-ray image
            patient_info: Optional patient demographics

        Returns:
            Structured ChestXrayReport
        """
        # Build context-aware prompt
        prompt = """Analyze this chest X-ray and provide a structured radiology report.

Please format your response as follows:

**FINDINGS:**
[Describe all observations including cardiac silhouette, lung fields, mediastinum, bones, and soft tissues]

**IMPRESSION:**
[Provide a concise summary of the key findings]

**DIFFERENTIAL DIAGNOSIS:**
[List possible diagnoses in order of likelihood]

**RECOMMENDATIONS:**
[Suggest any follow-up imaging or clinical correlation needed]
"""
        if patient_info:
            context = f"Patient: {patient_info.get('age', 'Unknown')} year old {patient_info.get('gender', 'patient')}"
            prompt = context + "\n\n" + prompt

        response = self.model.analyze_image(image, prompt, max_new_tokens=2000)

        # Parse the response (basic parsing)
        return ChestXrayReport(
            findings=self._extract_section(response, "FINDINGS"),
            impression=self._extract_section(response, "IMPRESSION"),
            differential=self._extract_list(response, "DIFFERENTIAL"),
            recommendations=self._extract_list(response, "RECOMMENDATIONS"),
            raw_response=response,
        )

    def get_differential_diagnosis(
        self,
        image: Union[Image.Image, str, Path],
        clinical_history: Optional[str] = None,
    ) -> List[str]:
        """
        Get differential diagnosis for an image.

        Args:
            image: Medical image
            clinical_history: Optional clinical context

        Returns:
            List of possible diagnoses
        """
        prompt = "Based on this image, provide a differential diagnosis. List the most likely diagnoses in order of probability."

        if clinical_history:
            prompt = f"Clinical history: {clinical_history}\n\n{prompt}"

        response = self.model.analyze_image(image, prompt)
        return self._parse_differential(response)

    def classify(
        self,
        image: Union[Image.Image, str, Path],
        labels: List[str],
    ) -> Dict[str, float]:
        """
        Zero-shot classification using MedSigLIP.

        Args:
            image: Medical image
            labels: List of possible labels

        Returns:
            Dictionary of label probabilities
        """
        if not self.classifier:
            raise RuntimeError("Classifier not loaded. Initialize with load_classifier=True")

        return self.classifier.classify(image, labels)

    def batch_analyze(
        self,
        images: List[Union[Image.Image, str, Path]],
        analysis_type: AnalysisType = AnalysisType.DESCRIBE,
    ) -> List[AnalysisResult]:
        """
        Analyze multiple images.

        Args:
            images: List of images
            analysis_type: Type of analysis

        Returns:
            List of AnalysisResults
        """
        results = []
        for i, image in enumerate(images):
            print(f"Analyzing image {i+1}/{len(images)}...")
            result = self.analyze(image, analysis_type)
            results.append(result)
        return results

    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from structured response."""
        lines = text.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            if section.upper() in line.upper() and ("**" in line or ":" in line):
                in_section = True
                continue
            elif in_section and line.strip().startswith("**"):
                break
            elif in_section and line.strip():
                section_lines.append(line.strip())

        return "\n".join(section_lines)

    def _extract_list(self, text: str, section: str) -> List[str]:
        """Extract a list from a section."""
        section_text = self._extract_section(text, section)
        items = []
        for line in section_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                # Remove bullet or number
                clean = line.lstrip("-•0123456789.) ").strip()
                if clean:
                    items.append(clean)
        return items if items else [section_text] if section_text else []

    def _parse_differential(self, text: str) -> List[str]:
        """Parse differential diagnosis from text."""
        diagnoses = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or
                        line[0].isdigit() or ":" in line):
                clean = line.lstrip("-•0123456789.) ").strip()
                if clean and len(clean) > 3:
                    diagnoses.append(clean)
        return diagnoses[:10]  # Limit to top 10


# =============================================================================
# Quick Analysis Functions
# =============================================================================

def analyze_chest_xray(
    image: Union[Image.Image, str, Path],
    device: str = "cuda",
) -> str:
    """Quick chest X-ray analysis."""
    analyzer = MedicalImageAnalyzer(device=device)
    result = analyzer.analyze(image, AnalysisType.FINDINGS)
    return result.response


def generate_xray_report(
    image: Union[Image.Image, str, Path],
    device: str = "cuda",
) -> ChestXrayReport:
    """Generate chest X-ray report."""
    analyzer = MedicalImageAnalyzer(device=device)
    return analyzer.generate_report(image)


def ask_medical_question(
    question: str,
    device: str = "cuda",
) -> str:
    """Answer a medical question."""
    model = MedGemma(device=device)
    return model.ask(question)


# =============================================================================
# Demo Pipeline
# =============================================================================

def run_demo(device: str = "cuda"):
    """
    Run a demo of the inference pipeline.

    This will:
    1. Load sample chest X-rays
    2. Analyze each with MedGemma
    3. Print results
    """
    print("=" * 60)
    print("MedGemma Inference Pipeline Demo")
    print("=" * 60)

    print("\nLoading model...")
    analyzer = MedicalImageAnalyzer(device=device)

    print("\nFetching sample chest X-rays...")
    samples = get_sample_chest_xrays(n=3)

    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: Labels = {sample['labels']}")
        print("=" * 60)

        result = analyzer.analyze(sample["image"], AnalysisType.FINDINGS)
        print(f"\nMedGemma Analysis:\n{result.response}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        print("Medical Image Analyzer")
        print("Usage: python -m src.inference --demo")
        print("\nOr import and use in your code:")
        print("  from src.inference import MedicalImageAnalyzer")
        print("  analyzer = MedicalImageAnalyzer()")
        print("  result = analyzer.analyze(image, AnalysisType.REPORT)")
