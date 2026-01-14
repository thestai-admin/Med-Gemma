# MedGemma Impact Challenge

Competition entry for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) - a Kaggle hackathon with $100,000 in prizes.

## Competition Overview

**Goal**: Build human-centered AI applications using MedGemma and other open models from Google's Health AI Developer Foundations (HAI-DEF).

**Deadline**: February 24, 2026

## Quick Start

### Prerequisites
- Python 3.10+
- Hugging Face account with HAI-DEF terms accepted
- GPU access (Kaggle notebooks or Google Colab)

### Installation

```bash
pip install -r requirements.txt
```

### Using MedGemma

```python
from transformers import pipeline
import torch

# Load the model
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# Analyze a chest X-ray
from PIL import Image
image = Image.open("path/to/xray.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this chest X-ray"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=2000)
print(output[0]["generated_text"][-1]["content"])
```

## Project Structure

```
Med Gemma/
├── CLAUDE.md           # Project context for Claude Code
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
├── app/                # Demo application
├── tests/              # Tests
├── data/               # Sample data
├── outputs/            # Generated outputs
└── submission/         # Competition submission materials
```

## Resources

- [MedGemma Model](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)
- [NIH Chest X-ray Dataset](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)
- [HAI-DEF Documentation](https://developers.google.com/health-ai-developer-foundations)

## License

This project uses models governed by the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).

## Disclaimer

This is a competition project. Model outputs are for demonstration purposes only and require clinical verification. Not intended for direct patient care.
