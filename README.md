# PrimaCare AI - MedGemma Impact Challenge

**Multi-agent diagnostic support system for primary care physicians**

Competition entry for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) - a Kaggle hackathon with $100,000 in prizes.

## Project Overview

PrimaCare AI is a multimodal diagnostic support system built on MedGemma 1.5 4B and MedSigLIP. It uses an agentic architecture to help primary care physicians:

- Structure patient histories into formal HPI format
- Analyze chest X-rays with zero-shot classification
- Generate differential diagnoses and recommended workups

### Architecture

```
Patient → IntakeAgent → ImagingAgent → ReasoningAgent → Clinical Output
              ↓              ↓              ↓
         Structured       X-ray        Differential
           HPI          Analysis        + Workup
```

## Quick Start

### Prerequisites
- Python 3.10+
- Hugging Face account with [HAI-DEF terms](https://huggingface.co/google/medgemma-1.5-4b-it) accepted
- GPU access (Kaggle notebooks or Google Colab with T4/P100)

### Run on Kaggle

1. Upload `notebooks/04-agentic-workflow.ipynb` to Kaggle
2. Add your HF_TOKEN as a Kaggle secret
3. Enable GPU accelerator (T4)
4. Run all cells

### Using MedGemma (Direct Model)

```python
# IMPORTANT: Disable torch dynamo BEFORE importing torch
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")

# Analyze a chest X-ray
image = Image.open("path/to/xray.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this chest X-ray"}
    ]
}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to("cuda")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=2000, do_sample=False)

generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01-model-exploration.ipynb` | MedGemma + MedSigLIP capabilities exploration |
| `03-prototype.ipynb` | PrimaCare AI full diagnostic pipeline |
| `04-agentic-workflow.ipynb` | Multi-agent system + Gradio demo (main submission) |

## Project Structure

```
Med Gemma/
├── notebooks/           # Kaggle-ready Jupyter notebooks
├── src/
│   ├── model.py        # MedGemma wrapper
│   ├── data.py         # Data loading utilities
│   ├── inference.py    # Inference pipeline
│   └── agents/         # Agent implementations
│       ├── intake.py
│       ├── imaging.py
│       ├── reasoning.py
│       └── orchestrator.py
├── app/
│   └── demo.py         # Gradio demo application
├── submission/
│   ├── writeup.md      # Competition writeup
│   └── video/          # Video demo script
└── requirements.txt    # Python dependencies
```

## Competition Details

- **Prize Pool**: $100,000
- **Deadline**: February 24, 2026
- **Target Tracks**:
  - Main Track ($10K-$30K)
  - Agentic Workflow Prize ($5K)

## Resources

- [MedGemma Model](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)
- [HAI-DEF Documentation](https://developers.google.com/health-ai-developer-foundations)
- [Competition Page](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

## Disclaimer

This is a competition project. Model outputs are for demonstration purposes only and require clinical verification. Not intended for direct patient care.

## License

This project uses models governed by the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
