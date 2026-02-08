# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PrimaCare AI is a multi-agent diagnostic support system for primary care physicians, built for the MedGemma Impact Challenge (Kaggle competition, deadline Feb 24, 2026). The submission narrative is CXR-first (chest X-ray analysis). It uses MedGemma 1.5 4B for multimodal analysis and MedSigLIP for zero-shot classification. Competes in all 4 award tracks: Main, Agentic Workflow, Novel Task (patient education), and Edge AI.

## Architecture

```
Patient -> IntakeAgent -> ImagingAgent -> ReasoningAgent -> GuidelinesAgent -> EducationAgent -> Output
              |               |               |                 |                  |
         Structured       X-ray         Differential      Evidence-Based     Patient-Friendly
         HPI + Flags    Analysis         + Workup        Recommendations      Education
```

**Tiered deployment:**
- Edge (CPU): MedSigLIP ONNX INT8 for binary pneumonia screening (`src/edge/`)
- Cloud (GPU): Full 5-agent pipeline with MedGemma

**Entry point:** `PrimaCareOrchestrator.run()` in `src/agents/orchestrator.py`. The orchestrator lazy-loads all models and agents via `@property` accessors — no model loads at import time.

**Agent pipeline (src/agents/):**
- `IntakeAgent` — Structures patient history into formal HPI format
- `ImagingAgent` — CXR analysis with MedGemma + zero-shot classification via MedSigLIP. Supports three classification modes: `multilabel`, `binary`, `ensemble`
- `ReasoningAgent` — Generates differential diagnosis, workup, and disposition
- `GuidelinesAgent` — RAG over clinical guidelines using sentence-transformers (CPU) + MedGemma synthesis
- `PatientEducationAgent` — Converts technical reports to patient-friendly language at 3 reading levels (basic, intermediate, detailed) with glossary

**Edge module (src/edge/):**
- `EdgeClassifier` — CPU-only pneumonia screening via ONNX-quantized MedSigLIP
- `quantize.py` — ONNX export + INT8 dynamic quantization
- `benchmark.py` — Latency, memory, and accuracy benchmarking

**Model wrappers (src/model.py):**
- `MedGemma` — Wraps `pipeline("image-text-to-text")` for local use. On Kaggle, use direct `AutoModelForImageTextToText` loading instead (see Kaggle section)
- `MedSigLIP` — CLIP-style zero-shot classification via `AutoModel`

**Data flow — structured dataclasses chained between agents:**
```
PatientContext -> ImageAnalysis -> ClinicalRecommendation -> GuidelinesResult -> PatientEducation -> PrimaCareResult
```
Each intermediate result has `to_prompt_context()` for chaining. `PrimaCareResult.to_report()` generates the final clinical report; `.to_dict()` returns JSON-serializable output.

**Lazy imports:** `src/__init__.py` uses `__getattr__` to lazily load all heavy submodules. Lightweight imports don't pull in torch/transformers.

**Evaluation (src/eval/):** Deterministic binary classification metrics (`confusion_counts`, `compute_binary_metrics`, `evaluate_scores`, `sweep_thresholds`, `select_threshold`, `bootstrap_metric_ci`) and latency profiling (`profile_orchestrator_latency`).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (all tests use mocks, no GPU needed)
pytest

# Run a specific test file or by keyword
pytest tests/test_education.py
pytest tests/test_edge.py
pytest -k longitudinal

# Run Gradio demo locally (requires GPU)
python app/demo.py

# Export edge model (requires GPU for initial export)
python scripts/export_edge_model.py

# Run edge benchmarks (CPU only)
python scripts/run_edge_benchmark.py

# Generate guideline embeddings (one-time setup, requires sentence-transformers)
python scripts/prepare_guidelines.py
```

## Development Environment

**Primary development happens on Kaggle notebooks** (T4 GPU). The recommended submission path is `notebooks/05-cxr-first-submission.ipynb`. `notebooks/04-agentic-workflow.ipynb` is the extended demo with profiling.

**Local development** is for code organization, tests, and git management — no GPU available locally. Tests use `mock_medgemma` and `mock_medsiglip` fixtures from `tests/conftest.py`.

## Kaggle-Specific Requirements

**T4 GPU constraints (~16GB VRAM):** MedGemma ~10GB + MedSigLIP ~4GB + sentence-transformers on CPU.

Every notebook must start with:
```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # MUST be before torch import

import torch
import warnings
warnings.filterwarnings('ignore')
```

Model loading on Kaggle (use direct loading, NOT pipeline):
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
```

HuggingFace auth on Kaggle:
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

For longitudinal imaging (multiple images), process sequentially and call `torch.cuda.empty_cache()` between analyses to avoid OOM.

## Testing

- Framework: pytest with shared fixtures in `tests/conftest.py`
- All tests run locally without GPU using `mock_medgemma`/`mock_medsiglip` fixtures
- Custom marker: `requires_gpu` (auto-skipped when GPU unavailable)
- Test naming: `tests/test_<feature>.py`, functions as `test_<behavior>()`
- Current: 42 tests passing, 1 skipped (GPU)

## Known Issues to Ignore

- `MessageFactory` warnings — protobuf version mismatch, harmless
- `slow image processor` warning — expected, works correctly

## External Resources

- MedGemma: `google/medgemma-1.5-4b-it` on HuggingFace
- MedSigLIP: `google/medsiglip-448` on HuggingFace
- HAI-DEF Docs: https://developers.google.com/health-ai-developer-foundations
- Competition: https://www.kaggle.com/competitions/med-gemma-impact-challenge
