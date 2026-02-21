# PrimaCare AI: Multi-Agent CXR Diagnostic Support with Patient Education and Edge Deployment

## Team
Solo submission

## Problem Statement

Primary care physicians face three interconnected challenges:

1. **CXR interpretation bottleneck**: Immediate radiology support is often unavailable, delaying diagnosis for conditions like pneumonia where early treatment improves outcomes.
2. **Health literacy gap**: ~36% of US adults have limited health literacy (NAAL). Technical radiology reports are incomprehensible to most patients, leading to worse adherence and outcomes.
3. **Resource-limited settings**: Rural and underserved clinics lack GPU infrastructure for AI-assisted diagnostics, creating care disparities.

PrimaCare AI addresses all three with a unified system: a 5-agent diagnostic pipeline, patient education at multiple reading levels, and an edge-deployable screening classifier.

## Solution Overview

### Architecture: 5-Agent Pipeline

```
Patient -> IntakeAgent -> ImagingAgent -> ReasoningAgent -> GuidelinesAgent -> EducationAgent -> Report
              |               |               |                 |                  |
         Structured       CXR Analysis    Differential     Evidence-Based     Patient-Friendly
         HPI + Flags    + Classification   + Workup       Recommendations      Education
```

**Orchestrator** (`PrimaCareOrchestrator`) coordinates all agents with:
- Lazy model loading and model sharing across agents
- Parallel execution mode (intake + imaging concurrent)
- Per-stage profiling with `profile=True`
- Fast mode for reduced latency (skips guidelines, switches to binary classification)

### HAI-DEF Models Used

| Model | Role | Deployment |
|-------|------|------------|
| MedGemma 1.5 4B | Image analysis, reasoning, education, guidelines synthesis | GPU (T4) |
| MedSigLIP 448 | Zero-shot CXR classification (multilabel, binary, ensemble) | GPU or Edge (ONNX) |
| all-MiniLM-L6-v2 | Guidelines embedding for RAG retrieval | CPU |

### Tiered Deployment: Edge Screening + Cloud Analysis

```
[Edge Tier - CPU Only]                    [Cloud Tier - GPU]
MedSigLIP ONNX INT8 ──> Pneumonia? ──Y──> Full 5-Agent Pipeline
       |                                        |
       └──── Normal (high confidence) ──> Done  └──> Full Report + Education
```

## Technical Details

### Agentic Workflow (5 Agents)

1. **IntakeAgent**: Structures free-text history into formal HPI with red flag detection and urgency scoring
2. **ImagingAgent**: MedGemma systematic CXR analysis + MedSigLIP classification with 3 modes (multilabel, binary, ensemble)
3. **ReasoningAgent**: Differential diagnosis, workup recommendations, disposition, and risk stratification
4. **GuidelinesAgent**: RAG over clinical practice guidelines using semantic retrieval (sentence-transformers) with keyword fallback
5. **PatientEducationAgent** (Novel): Converts technical reports into patient-friendly language at 3 reading levels (basic/6th grade, intermediate, detailed) with medical term glossary

### Novel Task: Patient Education

The PatientEducationAgent addresses health literacy by:
- Translating clinical reports into 3 reading levels matching diverse patient needs
- Generating structured output: simplified diagnosis, what it means, next steps, when to seek help
- Building a glossary of medical terms with plain-language definitions
- Computing **Flesch-Kincaid grade level** for each output, quantifying readability
- Integrating into the pipeline via `include_education=True` or standalone via the Gradio demo

### Edge AI: CPU-Only Screening

The edge module enables pneumonia screening without GPU:
- **Export**: MedSigLIP vision encoder exported to ONNX via `torch.onnx.export()`
- **Quantize**: Two strategies — standard INT8 (74% size reduction demo) and **selective INT8** (`quantize_onnx_selective_int8()`) that excludes attention/normalization nodes to preserve accuracy
- **Inference**: `EdgeClassifier` runs on CPU with pre-computed text embeddings
- **API parity**: `classify_pneumonia()` returns same format as `MedSigLIP.classify()`

## Quantitative Results

### Binary Pneumonia Classification (MedSigLIP, 100 samples)

| Mode | Accuracy | Precision | Recall | Specificity | F1 | AUROC |
|------|----------|-----------|--------|-------------|-----|-------|
| 10-label | 53.0% | 71.4% | 10.0% | 96.0% | 0.175 | — |
| **Binary** | **76.0%** | **68.1%** | **98.0%** | **54.0%** | **0.73** | **reported** |

AUROC is computed via `compute_auroc()` in `src/eval/cxr_eval.py` (pure numpy, no sklearn). ROC and Precision-Recall curves are plotted in the submission notebook.

### Pipeline Latency (Kaggle T4)

| Stage | Time |
|-------|------|
| Intake | 23.3s |
| Imaging | 16.9s |
| Reasoning | 38.0s |
| Guidelines | 32.9s |
| **Total** | **111.0s** |

Suitable for asynchronous decision support. Fast mode reduces latency by skipping guidelines and using binary classification.

## Evaluation Methodology

- Deterministic evaluation utilities in `src/eval/cxr_eval.py`
- Confusion matrix, binary metrics, threshold sweeps, bootstrap CIs, AUROC (`compute_auroc`)
- ROC curve and Precision-Recall curve plotted in the submission notebook
- Flesch-Kincaid grade level computed for each PatientEducation output
- Edge benchmarking with latency and memory profiling (`src/edge/benchmark.py`)
- All tests run locally without GPU (50 passed, 1 skipped)

## Limitations

- Evaluation is small-sample (100 images); larger validation recommended
- Specificity-recall tradeoff requires clinical objective selection
- Education readability validated by FK grade; formal user study recommended for clinical deployment
- Selective INT8 reduces accuracy degradation but FP32 ONNX is used for production edge inference

## Resources

- Submission notebook: `notebooks/05-cxr-first-submission.ipynb`
- Core code: `src/agents/`, `src/edge/`, `src/eval/`
- Demo: `app/demo.py` (7 tabs including education)
- Repository: Public on GitHub
