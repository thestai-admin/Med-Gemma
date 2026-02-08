"""
HAI-DEF Model Wrappers

Provides clean interfaces for working with Google Health AI Developer Foundations models:
- MedGemma: Multimodal medical AI (text + images)
- MedSigLIP: Medical image encoder (classification, retrieval)
- MedASR: Medical speech recognition

Official docs: https://developers.google.com/health-ai-developer-foundations
"""

import torch
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModel,
)
from PIL import Image
from typing import Union, List, Dict, Any, Optional
from pathlib import Path


# =============================================================================
# Model IDs
# =============================================================================

class ModelIDs:
    """Official Hugging Face model IDs for HAI-DEF models."""

    # MedGemma variants
    MEDGEMMA_1_5_4B = "google/medgemma-1.5-4b-it"  # Latest, recommended
    MEDGEMMA_4B_IT = "google/medgemma-4b-it"
    MEDGEMMA_4B_PT = "google/medgemma-4b-pt"
    MEDGEMMA_27B_TEXT = "google/medgemma-27b-text-it"
    MEDGEMMA_27B_IT = "google/medgemma-27b-it"

    # MedSigLIP
    MEDSIGLIP_448 = "google/medsiglip-448"

    # MedASR
    MEDASR = "google/medasr"


# =============================================================================
# MedGemma - Multimodal Medical AI
# =============================================================================

class MedGemma:
    """
    Wrapper for MedGemma models.

    MedGemma is a Gemma 3 variant trained for medical text and image comprehension.
    Supports chest X-rays, CT, MRI, dermatology, histopathology, and more.

    Usage:
        model = MedGemma()
        response = model.analyze_image(image, "Describe this chest X-ray")
    """

    def __init__(
        self,
        model_id: str = ModelIDs.MEDGEMMA_1_5_4B,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize MedGemma model.

        Args:
            model_id: Hugging Face model ID (default: latest 1.5 4B)
            device: Device to run on ("cuda" or "cpu")
            torch_dtype: Data type for model weights (bfloat16 recommended)
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
        )

    def analyze_image(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 2000,
    ) -> str:
        """
        Analyze a medical image with a text prompt.

        Args:
            image: PIL Image, file path, or URL
            prompt: Text prompt for analysis
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        output = self.pipe(text=messages, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"][-1]["content"]

    def ask(
        self,
        question: str,
        max_new_tokens: int = 1000,
    ) -> str:
        """
        Ask a text-only medical question.

        Args:
            question: Medical question text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]
        output = self.pipe(text=messages, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"][-1]["content"]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2000,
    ) -> str:
        """
        Multi-turn conversation.

        Args:
            messages: List of message dicts with "role" and "content"
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        # Normalize plain-string content to list-of-dicts format
        # required by newer transformers image-text-to-text pipeline
        normalized = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": msg["role"], "content": content})

        output = self.pipe(text=normalized, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"][-1]["content"]


# =============================================================================
# MedSigLIP - Medical Image Encoder
# =============================================================================

class MedSigLIP:
    """
    Wrapper for MedSigLIP model.

    MedSigLIP is a medical image encoder for zero-shot classification and retrieval.
    Use this for classification tasks; use MedGemma for text generation.

    Usage:
        model = MedSigLIP()
        probs = model.classify(image, ["normal", "pneumonia", "effusion"])
    """

    def __init__(
        self,
        model_id: str = ModelIDs.MEDSIGLIP_448,
        device: str = "cuda",
    ):
        """
        Initialize MedSigLIP model.

        Args:
            model_id: Hugging Face model ID
            device: Device to run on
        """
        self.model_id = model_id
        self.device = device

        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def classify(
        self,
        image: Union[Image.Image, str, Path],
        labels: List[str],
    ) -> Dict[str, float]:
        """
        Zero-shot classification of medical image.

        Args:
            image: PIL Image or file path
            labels: List of text labels to classify against

        Returns:
            Dictionary mapping labels to probabilities
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        inputs = self.processor(
            text=labels,
            images=[image],
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits_per_image, dim=1)[0]

        return {label: prob.item() for label, prob in zip(labels, probs)}

    def get_embeddings(
        self,
        images: List[Union[Image.Image, str, Path]],
    ) -> torch.Tensor:
        """
        Get image embeddings for retrieval tasks.

        Args:
            images: List of PIL Images or file paths

        Returns:
            Image embeddings tensor
        """
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img))
            else:
                pil_images.append(img)

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return outputs


# =============================================================================
# Prompt Templates
# =============================================================================

PROMPTS = {
    # General
    "describe": "Describe this medical image in detail.",
    "findings": "List all findings in this image in bullet points.",
    "differential": "What is your differential diagnosis based on this image?",
    "quality": "Assess the technical quality of this image.",

    # Chest X-ray specific
    "cxr_report": """Generate a structured radiology report for this chest X-ray:
1. Technique
2. Comparison (if prior available)
3. Findings
4. Impression""",

    "cxr_findings": "Describe the cardiac silhouette, lung fields, and any abnormalities.",

    # Primary care
    "primary_care": """As a primary care physician reviewing this image:
1. Key findings
2. Differential diagnosis
3. Recommended next steps""",
}


# =============================================================================
# Convenience Functions
# =============================================================================

def get_medgemma(
    model_id: str = ModelIDs.MEDGEMMA_1_5_4B,
    device: str = "cuda",
) -> MedGemma:
    """Get a MedGemma model instance."""
    return MedGemma(model_id=model_id, device=device)


def get_medsiglip(device: str = "cuda") -> MedSigLIP:
    """Get a MedSigLIP model instance."""
    return MedSigLIP(device=device)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test MedGemma
    print("Loading MedGemma...")
    model = MedGemma()

    # Test with a sample prompt
    response = model.ask("What are the common findings on a chest X-ray for pneumonia?")
    print(f"Response: {response[:500]}...")
