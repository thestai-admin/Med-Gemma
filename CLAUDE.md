# MedGemma Impact Challenge Project

## Project: PrimaCare AI
**Multi-agent diagnostic support system for primary care physicians**

## Competition Details
- **Competition**: [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- **Deadline**: February 24, 2026 (11:59 PM UTC)
- **Prize Pool**: $100,000 (1st: $30K, 2nd: $20K, 3rd: $15K, 4th: $10K)
- **Special Prizes**: Agentic Workflow ($5K), Novel Task ($5K), Edge AI ($5K)
- **Target**: Main Track + Agentic Workflow Prize

---

## MASTER CHECKLIST

### Phase 1: Foundation (COMPLETED)
- [x] Project structure created
- [x] MedGemma 1.5 4B working on Kaggle T4
- [x] MedSigLIP zero-shot classification working
- [x] Basic notebooks (01, 03, 04) validated
- [x] Agentic architecture implemented (4 agents)
- [x] Gradio demo working with public URL
- [x] GitHub repo pushed
- [x] Basic writeup created
- [x] Video script outline created

### Phase 2: Improvements (IN PROGRESS - Week 1-2)

#### High Priority
- [ ] **Submit notebook to Kaggle** (get early visibility)
- [ ] **NIH ChestX-ray14 evaluation** - Test on real dataset, add metrics
- [ ] **Add more demo cases** - Pneumothorax, CHF, lung mass, TB
- [ ] **Improve prompts** - Better structured outputs from agents

#### Medium Priority
- [ ] **LoRA fine-tuning** - Fine-tune on chest X-ray findings (Novel Task Prize)
- [ ] **Add accuracy metrics** - Calculate precision/recall on test set
- [ ] **Error handling** - Handle edge cases gracefully
- [ ] **Polish Gradio UI** - Better layout, loading indicators

#### Low Priority
- [ ] **CT scan support** - Extend to 3D imaging
- [ ] **MedASR integration** - Voice input support
- [ ] **Batch processing** - Multiple images at once

### Phase 3: Submission Materials (Week 3-4)
- [ ] **Record 3-min video** - Use script in submission/video/
- [ ] **Finalize writeup** - Polish 3-page document
- [ ] **Create Kaggle submission** - Final notebook with outputs
- [ ] **Public demo** - Ensure Gradio URL is accessible

### Phase 4: Final Polish (Week 5-6)
- [ ] **Review against criteria** - Score each criterion
- [ ] **Get feedback** - Share with colleagues
- [ ] **Final submission** - Before Feb 24, 11:59 PM UTC
- [ ] **Backup submission** - Download all materials

---

## Evaluation Criteria Scorecard

| Criteria | Weight | Our Status | Target |
|----------|--------|------------|--------|
| Execution & Communication | 30% | Video/writeup pending | Polish video |
| Effective HAI-DEF Model Use | 20% | MedGemma + MedSigLIP | Add fine-tuning |
| Product Feasibility | 20% | Working demo | Add metrics |
| Problem Domain | 15% | Primary care focus | More cases |
| Impact Potential | 15% | Good narrative | Quantify impact |

---

## Current Architecture

```
Patient → IntakeAgent → ImagingAgent → ReasoningAgent → Output
              ↓              ↓              ↓
         Structured       X-ray        Differential
         HPI + Flags    Analysis        + Workup
```

**Agents:**
1. **IntakeAgent** - Structures HPI, identifies red flags
2. **ImagingAgent** - MedGemma analysis + MedSigLIP classification
3. **ReasoningAgent** - Differential diagnosis + workup
4. **Orchestrator** - Coordinates pipeline

---

## Technical Notes (Kaggle)

### Required First Cell
```python
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import warnings
warnings.filterwarnings('ignore')
```

### Model Loading (Direct - NOT pipeline)
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it")
```

### HF Login (Kaggle)
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
```

### Known Issues (Ignore)
- `MessageFactory` warnings - protobuf version, doesn't affect execution
- `slow image processor` warning - expected, works fine

---

## Files Reference

| File | Purpose |
|------|---------|
| `notebooks/04-agentic-workflow.ipynb` | **Main submission notebook** |
| `notebooks/03-prototype.ipynb` | PrimaCare AI pipeline |
| `notebooks/01-model-exploration.ipynb` | Model exploration |
| `submission/writeup.md` | 3-page competition writeup |
| `submission/video/video_script.md` | Video recording script |
| `src/agents/` | Agent implementations |
| `app/demo.py` | Gradio demo app |

---

## Datasets

### Currently Using
- `hf-vision/chest-xray-pneumonia` - Simple, works on Kaggle

### To Add
- `alkzar90/NIH-Chest-X-ray-dataset` - 112K images, 14 pathologies
- More diverse pathology cases

---

## Next Session Tasks

1. Submit `04-agentic-workflow.ipynb` to Kaggle (public)
2. Add NIH ChestX-ray14 evaluation cell
3. Add 3-4 more clinical demo cases
4. Calculate accuracy metrics

---

## Important Links

| Resource | URL |
|----------|-----|
| Competition | https://www.kaggle.com/competitions/med-gemma-impact-challenge |
| Our GitHub | https://github.com/thestai-admin/Med-Gemma |
| MedGemma Model | https://huggingface.co/google/medgemma-1.5-4b-it |
| MedSigLIP Model | https://huggingface.co/google/medsiglip-448 |
| NIH Dataset (Kaggle) | https://www.kaggle.com/datasets/nih-chest-xrays/data |
| HAI-DEF Docs | https://developers.google.com/health-ai-developer-foundations |

---

## Quick Commands

```bash
# Push to GitHub
cd "/mnt/c/Users/tarke/Desktop/Med Gemma"
git add -A && git commit -m "message" && git push

# Check status
git status
```

---

## Notes

- **Deadline**: Feb 24, 2026 (~6 weeks remaining)
- **GPU**: Kaggle T4 (30hrs/week free)
- **Submission**: Can update until deadline
- **Medals**: Based on community votes (already better than silver notebook)
- **Prizes**: Based on judge evaluation criteria
