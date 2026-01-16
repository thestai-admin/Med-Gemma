# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PrimaCare AI is a multi-agent diagnostic support system for primary care physicians, built for the MedGemma Impact Challenge (Kaggle competition, deadline Feb 24, 2026). It uses MedGemma 1.5 4B and MedSigLIP for chest X-ray analysis and clinical reasoning.

## Architecture

```
Patient → IntakeAgent → ImagingAgent → ReasoningAgent → Clinical Output
              ↓              ↓              ↓
         Structured       X-ray        Differential
         HPI + Flags    Analysis        + Workup
```

**Agent Pipeline (src/agents/):**
- `IntakeAgent` - Structures patient history into formal HPI format using MedGemma text generation
- `ImagingAgent` - Analyzes chest X-rays with MedGemma + zero-shot classification with MedSigLIP
- `ReasoningAgent` - Generates differential diagnosis, workup recommendations, and disposition
- `PrimaCareOrchestrator` - Coordinates all agents, manages model sharing via lazy loading

**Model Wrappers (src/model.py):**
- `MedGemma` - Wrapper using `pipeline("image-text-to-text")` for multimodal analysis
- `MedSigLIP` - Zero-shot classification using CLIP-style image-text matching

## Commands

```bash
# Run Gradio demo locally
python app/demo.py

# Push to GitHub
git add -A && git commit -m "message" && git push
```

## Development Environment

**Primary development happens on Kaggle notebooks** (T4 GPU, 30hrs/week free). The `notebooks/04-agentic-workflow.ipynb` is the main submission notebook.

**Local development** is for code organization and git management only - no GPU available locally.

## Kaggle-Specific Requirements

Every notebook must start with:
```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # MUST be before torch import

import torch
import warnings
warnings.filterwarnings('ignore')
```

Model loading (use direct loading, NOT pipeline, for Kaggle compatibility):
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
```

HuggingFace authentication on Kaggle:
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

## Key Files

| File | Purpose |
|------|---------|
| `notebooks/04-agentic-workflow.ipynb` | Main competition submission |
| `notebooks/03-prototype.ipynb` | PrimaCare AI pipeline development |
| `src/agents/orchestrator.py` | Main entry point - `PrimaCareOrchestrator.run()` |
| `src/model.py` | MedGemma and MedSigLIP wrappers |
| `app/demo.py` | Gradio demo application |

## Data Classes

The agent pipeline uses these dataclasses to pass structured data:

- `PatientContext` / `StructuredHPI` (intake.py) - Patient demographics and HPI elements
- `ImageAnalysis` / `ImagingFinding` (imaging.py) - X-ray findings and classification results
- `ClinicalRecommendation` / `Diagnosis` / `WorkupItem` (reasoning.py) - Clinical output
- `PrimaCareResult` (orchestrator.py) - Complete pipeline result with `to_report()` method

## Known Issues to Ignore

- `MessageFactory` warnings - protobuf version mismatch, doesn't affect execution
- `slow image processor` warning - expected behavior, works correctly

## External Resources

- MedGemma: `google/medgemma-1.5-4b-it` on HuggingFace
- MedSigLIP: `google/medsiglip-448` on HuggingFace
- HAI-DEF Docs: https://developers.google.com/health-ai-developer-foundations
- Competition: https://www.kaggle.com/competitions/med-gemma-impact-challenge
