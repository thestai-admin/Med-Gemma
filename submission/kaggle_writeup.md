# PrimaCare AI: Multi-Agent CXR Diagnostic Support with Patient Education and Edge Deployment

## Team
Solo submission

## Award Tracks
**Main Track** | **Agentic Workflow Prize** | **Novel Task Prize** | **Edge AI Prize**

---

## Problem Statement

Primary care physicians face three interconnected challenges:

1. **CXR interpretation bottleneck** — Immediate radiology support is often unavailable, delaying diagnosis for conditions like pneumonia where early treatment improves outcomes.
2. **Health literacy gap** — ~36% of US adults have limited health literacy (NAAL). Technical radiology reports are incomprehensible to most patients, leading to worse adherence and outcomes.
3. **Resource-limited settings** — Rural and underserved clinics lack GPU infrastructure for AI-assisted diagnostics, creating care disparities.

PrimaCare AI addresses all three with a unified system.

---

## Solution: 5-Agent Pipeline + Edge Screening

### Architecture

```
Patient → IntakeAgent → ImagingAgent → ReasoningAgent → GuidelinesAgent → EducationAgent → Report
              ↓              ↓              ↓                ↓                  ↓
         Structured       CXR Analysis  Differential     Evidence-Based    Patient-Friendly
         HPI + Flags    + Classification  + Workup       Recommendations    Education
```

### Tiered Deployment

```
[Edge - CPU Only]                         [Cloud - GPU]
MedSigLIP ONNX INT8 → Pneumonia? ──Y──→ Full 5-Agent Pipeline → Report + Education
       │
       └── Normal (high confidence) ──→ Done
```

### HAI-DEF Models Used

| Model | Role | Deployment |
|-------|------|------------|
| **MedGemma 1.5 4B** | Image analysis, reasoning, education, guideline synthesis | GPU (T4) |
| **MedSigLIP 448** | Zero-shot CXR classification (multilabel, binary, ensemble) | GPU or Edge (ONNX) |
| all-MiniLM-L6-v2 | Guidelines RAG embedding | CPU |

---

## Track 1: Main Track — CXR Diagnostic Pipeline

The `PrimaCareOrchestrator` coordinates 5 agents through a structured pipeline:

- **IntakeAgent**: Structures free-text into formal HPI with red flag detection
- **ImagingAgent**: MedGemma systematic CXR analysis + MedSigLIP classification (3 modes)
- **ReasoningAgent**: Differential diagnosis, workup, disposition, risk stratification
- **GuidelinesAgent**: RAG with semantic retrieval + keyword fallback
- **EducationAgent**: Patient-friendly translation at 3 reading levels

### Quantitative Results (Binary Pneumonia, 100 samples)

| Metric | 10-Label | Binary Mode |
|--------|----------|-------------|
| Accuracy | 53.0% | **76.0%** |
| Precision | 71.4% | **68.1%** |
| Recall | 10.0% | **98.0%** |
| F1 | 0.175 | **0.803** |

Binary mode materially improves sensitivity. Threshold selection utilities and bootstrap CIs are included for transparent evaluation.

---

## Track 2: Agentic Workflow Prize

### 5-Agent Coordination

The orchestrator provides:
- **Lazy model loading** — Models load on first use, shared across agents
- **Parallel execution** — Intake and imaging run concurrently via ThreadPoolExecutor
- **Per-stage profiling** — `profile=True` captures timing per agent
- **Fast mode** — Skips guidelines, uses binary classification for lower latency
- **RAG** — GuidelinesAgent uses sentence-transformers for semantic retrieval with keyword fallback

### Pipeline Latency (Kaggle T4)

| Stage | Median Time |
|-------|-------------|
| Intake | 23.3s |
| Imaging | 16.9s |
| Reasoning | 38.0s |
| Guidelines | 32.9s |
| Education | ~15s |
| **Total** | **~126s** |

---

## Track 3: Novel Task Prize — Patient Education

### The Problem
~36% of US adults have limited health literacy. Technical radiology reports are incomprehensible to most patients.

### The Solution
`PatientEducationAgent` converts clinical reports into patient-friendly language at 3 reading levels:

| Level | Target Audience | Example |
|-------|----------------|---------|
| **Basic** | 6th-grade reading level | "You have a lung infection that is making it hard to breathe" |
| **Intermediate** | General adult | "Community-acquired pneumonia with consolidation in the right lower lobe" |
| **Detailed** | Patients wanting thorough understanding | Full clinical terminology with inline definitions |

### Structured Output
Each education output includes:
- **Simplified Diagnosis** — What was found
- **What It Means** — Clinical significance in accessible language
- **Next Steps** — Actionable follow-up
- **When to Seek Help** — Warning signs for immediate care
- **Glossary** — Medical terms defined in plain language

### Integration
```python
result = orchestrator.run(
    chief_complaint="Cough",
    xray_image=image,
    include_education=True,
    education_level="basic",
)
print(result.patient_education.to_report_section())
```

---

## Track 4: Edge AI Prize — CPU-Only Screening

### Motivation
Rural and underserved clinics often lack GPU infrastructure. Edge AI enables pneumonia screening on any laptop or desktop.

### Implementation
1. **Export**: MedSigLIP vision encoder → ONNX FP32 (`torch.onnx.export`)
2. **Quantize**: INT8 dynamic quantization (`onnxruntime.quantization`)
3. **Pre-compute**: Text embeddings for "normal" / "pneumonia" saved as `.npy`
4. **Inference**: Vision encoder on CPU + cosine similarity with cached embeddings

### EdgeClassifier API
```python
from src.edge import EdgeClassifier

classifier = EdgeClassifier("models/edge/medsiglip_int8.onnx")
result = classifier.classify_pneumonia(image)
# {"normal": 0.82, "pneumonia": 0.18}
```

API mirrors `MedSigLIP.classify()` for drop-in replacement.

### Benchmarking
Built-in benchmarking utilities measure latency, memory, and accuracy:
```python
from src.edge import run_edge_benchmark, compare_models
result = run_edge_benchmark(classifier, images, labels)
print(compare_models(gpu_result, edge_result))
```

---

## Evaluation & Reproducibility

- **Deterministic evaluation**: `src/eval/cxr_eval.py` — confusion counts, binary metrics, threshold sweeps, bootstrap CIs
- **Edge benchmarks**: `src/edge/benchmark.py` — latency, memory, accuracy profiling
- **Test suite**: 42 tests passing (all mock-based, no GPU needed)
- **Notebook**: `05-cxr-first-submission.ipynb` — end-to-end reproducible on Kaggle T4

## Limitations

- Evaluation is small-sample (100 images); larger validation needed
- Education quality is qualitative; formal readability testing recommended
- Edge accuracy may degrade with INT8; documented transparently
- Pipeline latency (~2 min) is suitable for async second-opinion, not real-time triage

---

## Links

- **Notebook**: [05-cxr-first-submission.ipynb](https://www.kaggle.com/code/YOUR_USERNAME/primacare-ai-submission) *(update with your Kaggle notebook URL)*
- **Code**: [github.com/thestai-admin/Med-Gemma](https://github.com/thestai-admin/Med-Gemma)
- **Video**: *(add your video link here)*
- **Demo**: `python app/demo.py` (7 tabs including Patient Education)

---

*PrimaCare AI is clinician decision support, not autonomous diagnosis. All outputs require verification by qualified healthcare professionals.*
