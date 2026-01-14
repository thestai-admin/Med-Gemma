"""
MedGemma Model Wrapper

Provides a clean interface for working with MedGemma 1.5 4B.
"""

import torch
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import Union, List, Dict, Any, Optional


class MedGemmaModel:
    """Wrapper for MedGemma 1.5 4B model."""

    MODEL_ID = "google/medgemma-1.5-4b-it"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_pipeline: bool = True
    ):
        """
        Initialize MedGemma model.

        Args:
            model_id: Hugging Face model ID
            device: Device to run on ("cuda" or "cpu")
            torch_dtype: Data type for model weights
            use_pipeline: If True, use simple pipeline API
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        if use_pipeline:
            self.pipe = pipeline(
                "image-text-to-text",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
            )
            self.model = None
            self.processor = None
        else:
            self.pipe = None
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

    def analyze_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 2000
    ) -> str:
        """
        Analyze a medical image with a text prompt.

        Args:
            image: PIL Image or path to image file
            prompt: Text prompt for analysis
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if isinstance(image, str):
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

        if self.pipe:
            output = self.pipe(text=messages, max_new_tokens=max_new_tokens)
            return output[0]["generated_text"][-1]["content"]
        else:
            return self._generate_with_model(messages, max_new_tokens)

    def ask(
        self,
        question: str,
        max_new_tokens: int = 1000
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
                "content": question
            }
        ]

        if self.pipe:
            output = self.pipe(text=messages, max_new_tokens=max_new_tokens)
            return output[0]["generated_text"][-1]["content"]
        else:
            return self._generate_with_model(messages, max_new_tokens)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2000
    ) -> str:
        """
        Multi-turn conversation.

        Args:
            messages: List of message dicts with "role" and "content"
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if self.pipe:
            output = self.pipe(text=messages, max_new_tokens=max_new_tokens)
            return output[0]["generated_text"][-1]["content"]
        else:
            return self._generate_with_model(messages, max_new_tokens)

    def _generate_with_model(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int
    ) -> str:
        """Generate using direct model API."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.torch_dtype)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)


# Common prompt templates
PROMPTS = {
    "describe": "Describe this medical image in detail.",
    "findings": "List all findings in this image in bullet points.",
    "differential": "What is your differential diagnosis based on this image?",
    "quality": "Assess the technical quality of this image.",
    "cxr_report": """Generate a structured radiology report for this chest X-ray including:
1. Technique
2. Comparison
3. Findings
4. Impression""",
}


def get_model(device: str = "cuda") -> MedGemmaModel:
    """Get a MedGemma model instance."""
    return MedGemmaModel(device=device)
