# PrimaCare AI - MedGemma Impact Challenge

**CXR-first multi-agent diagnostic support with patient education and edge deployment**

Competition entry for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) - a Kaggle hackathon with $100,000 in prizes.

## Project Overview

PrimaCare AI is a CXR-first diagnostic support system built on MedGemma 1.5 4B and MedSigLIP. It uses a 5-agent agentic architecture to help primary care physicians:

- Structure patient histories into formal HPI format
- Analyze chest X-rays with zero-shot classification
- Generate differential diagnoses and recommended workups
- Retrieve evidence-based clinical practice guidelines (RAG)
- **Translate findings into patient-friendly language at 3 reading levels**

For resource-limited settings, the edge module provides **CPU-only pneumonia screening** using ONNX-quantized MedSigLIP.

### Architecture

```
Patient -> IntakeAgent -> ImagingAgent -> ReasoningAgent -> GuidelinesAgent -> EducationAgent -> Output
              |               |               |                 |                  |
         Structured       X-ray         Differential      Evidence-Based     Patient-Friendly
         HPI + Flags    Analysis         + Workup        Recommendations      Education
```

### Tiered Deployment

```
[Edge - CPU Only]                         [Cloud - GPU]
MedSigLIP ONNX INT8 -> Pneumonia? --Y--> Full 5-Agent Pipeline
       |                                        |
       +--- Normal ----------------------> Done  +---> Report + Education
```

## Award Tracks

| Track | Prize | Status |
|-------|-------|--------|
| Main Track | $75K | 5-agent pipeline, F1 0.803 binary pneumonia |
| Agentic Workflow | $10K | 5 agents, orchestrator, RAG, profiling |
| Novel Task | $10K | PatientEducationAgent (3 reading levels + glossary) |
| Edge AI | $5K | MedSigLIP ONNX INT8, CPU-only inference |

## Quick Start

### Run on Kaggle

1. Upload `notebooks/05-cxr-first-submission.ipynb` to Kaggle
2. Add your HF_TOKEN as a Kaggle secret
3. Enable GPU accelerator (T4)
4. Run all cells

### Using MedGemma (Direct Model)

```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")

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

### Using the Pipeline

```python
from src.agents import PrimaCareOrchestrator

orchestrator = PrimaCareOrchestrator()
result = orchestrator.run(
    chief_complaint="Cough for 2 weeks with fever",
    history="65yo male smoker",
    xray_image=image,
    include_education=True,       # Generate patient education
    education_level="basic",      # 6th-grade reading level
    profile=True,                 # Capture timings
)
print(result.to_report())
```

### Edge AI (CPU-Only)

```python
from src.edge import EdgeClassifier

classifier = EdgeClassifier("models/edge/medsiglip_int8.onnx")
result = classifier.classify_pneumonia(image)
# {"normal": 0.82, "pneumonia": 0.18}
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (all use mocks, no GPU needed)
pytest

# Run Gradio demo locally (requires GPU)
python app/demo.py

# Export edge model (requires GPU for initial export)
python scripts/export_edge_model.py

# Run edge benchmarks (CPU only)
python scripts/run_edge_benchmark.py

# Generate guideline embeddings
python scripts/prepare_guidelines.py
```

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `05-cxr-first-submission.ipynb` | CXR-first reproducible submission with education + edge | **Primary** |
| `04-agentic-workflow.ipynb` | Extended demo with evaluation metrics and profiling | Reference |

## Project Structure

```
Med Gemma/
├── notebooks/              # Kaggle-ready Jupyter notebooks
│   ├── 05-cxr-first-submission.ipynb  # Primary submission
│   └── 04-agentic-workflow.ipynb      # Extended demo
├── src/
│   ├── model.py           # MedGemma + MedSigLIP wrappers
│   ├── eval/              # Reproducible CXR evaluation utilities
│   ├── agents/            # Agent implementations
│   │   ├── intake.py      # IntakeAgent - HPI structuring
│   │   ├── imaging.py     # ImagingAgent - X-ray analysis
│   │   ├── reasoning.py   # ReasoningAgent - Differential Dx
│   │   ├── guidelines.py  # GuidelinesAgent - RAG
│   │   ├── education.py   # PatientEducationAgent - Health literacy
│   │   └── orchestrator.py # PrimaCareOrchestrator
│   └── edge/              # Edge AI module
│       ├── inference.py   # EdgeClassifier (ONNX CPU)
│       ├── quantize.py    # ONNX export + INT8 quantization
│       └── benchmark.py   # Latency/accuracy benchmarks
├── data/
│   └── guidelines/        # Clinical guidelines for RAG
├── app/
│   └── demo.py            # Gradio demo (7 tabs)
├── scripts/
│   ├── prepare_guidelines.py    # Embedding generation
│   ├── export_edge_model.py     # ONNX export script
│   └── run_edge_benchmark.py    # Benchmark runner
├── submission/
│   ├── writeup.md         # Competition writeup (all 4 tracks)
│   └── video/             # Video demo materials
├── tests/                 # 42 tests, all mock-based
└── requirements.txt
```

## Deployment

### Requirements

- **GPU**: NVIDIA T4 (16GB VRAM) for full pipeline
- **CPU**: Any modern CPU for edge classifier
- **Memory**: 16GB+ system RAM
- **Python**: 3.10+

### Model Requirements

| Model | Size | VRAM | Deployment |
|-------|------|------|------------|
| MedGemma 1.5 4B | ~8GB | ~10GB | GPU only |
| MedSigLIP | ~3.5GB | ~4GB | GPU or Edge (ONNX) |
| MedSigLIP INT8 | ~500MB | CPU | Edge only |
| sentence-transformers | ~90MB | CPU | CPU |

## Resources

- [MedGemma Model](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)
- [HAI-DEF Documentation](https://developers.google.com/health-ai-developer-foundations)
- [Competition Page](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

## Disclaimer

This is a competition project. Model outputs are for demonstration purposes only and require clinical verification. Not intended for direct patient care.

## License

This project uses models governed by the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
