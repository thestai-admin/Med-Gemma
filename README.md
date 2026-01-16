# PrimaCare AI - MedGemma Impact Challenge

**Multi-agent diagnostic support system for primary care physicians**

Competition entry for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) - a Kaggle hackathon with $100,000 in prizes.

## Project Overview

PrimaCare AI is a multimodal diagnostic support system built on MedGemma 1.5 4B and MedSigLIP. It uses an agentic architecture to help primary care physicians:

- Structure patient histories into formal HPI format
- Analyze chest X-rays with zero-shot classification
- Generate differential diagnoses and recommended workups
- Retrieve evidence-based clinical practice guidelines (RAG)
- Generate patient-friendly explanations

### Architecture

```
Patient → IntakeAgent → ImagingAgent → ReasoningAgent → GuidelinesAgent → Output
              ↓              ↓              ↓                 ↓
         Structured       X-ray        Differential    Evidence-Based
           HPI          Analysis        + Workup       Recommendations
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

| Notebook | Description | Status |
|----------|-------------|--------|
| `04-agentic-workflow.ipynb` | **Main submission** - 4-agent pipeline with RAG, evaluation metrics, Gradio demo | **Primary** |
| `03-prototype.ipynb` | PrimaCare AI pipeline development | Reference |
| `01-model-exploration.ipynb` | MedGemma + MedSigLIP capabilities exploration | Reference |

## Project Structure

```
Med Gemma/
├── notebooks/              # Kaggle-ready Jupyter notebooks
│   └── 04-agentic-workflow.ipynb  # Main submission
├── src/
│   ├── model.py           # MedGemma wrapper
│   ├── data.py            # Data loading utilities
│   ├── inference.py       # Inference pipeline
│   └── agents/            # Agent implementations
│       ├── intake.py      # IntakeAgent - HPI structuring
│       ├── imaging.py     # ImagingAgent - X-ray analysis
│       ├── reasoning.py   # ReasoningAgent - Differential Dx
│       ├── guidelines.py  # GuidelinesAgent - RAG
│       └── orchestrator.py # PrimaCareOrchestrator
├── data/
│   └── guidelines/        # Clinical guidelines for RAG
│       └── chunks.json    # 47 guideline chunks
├── app/
│   └── demo.py            # Gradio demo application
├── scripts/
│   └── prepare_guidelines.py  # Embedding generation script
├── submission/
│   ├── writeup.md         # Competition writeup
│   └── video/             # Video demo materials
└── requirements.txt       # Python dependencies
```

## Deployment

### Requirements

- **GPU**: NVIDIA T4 (16GB VRAM) minimum, recommended for Kaggle
- **Memory**: 16GB+ system RAM
- **Storage**: ~15GB for models
- **Python**: 3.10+

### Option 1: Kaggle Notebooks (Recommended)

1. Upload `notebooks/04-agentic-workflow.ipynb` to Kaggle
2. Add your `HF_TOKEN` as a Kaggle secret
3. Enable GPU accelerator (T4 x2 or P100)
4. Run all cells

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/thestai-admin/Med-Gemma.git
cd Med-Gemma

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token
export HF_TOKEN=your_token_here

# Run Gradio demo
python app/demo.py
```

### Option 3: Google Colab

1. Open notebook in Colab
2. Change runtime to GPU (T4)
3. Add HF_TOKEN to Colab secrets
4. Run all cells

### Model Requirements

| Model | Size | VRAM |
|-------|------|------|
| MedGemma 1.5 4B | ~8GB | ~10GB |
| MedSigLIP | ~3.5GB | ~4GB |
| sentence-transformers | ~90MB | CPU |

**Total VRAM**: ~14GB (fits on T4 16GB)

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
