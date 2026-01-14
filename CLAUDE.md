# MedGemma Impact Challenge Project

## Project Overview
Competition entry for the MedGemma Impact Challenge (Kaggle hackathon).
Goal: Build the best human-centered AI application using MedGemma for healthcare.

## Competition Details
- **Competition**: [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- **Deadline**: February 24, 2026 (11:59 PM UTC)
- **Prize Pool**: $100,000 (1st place: $30,000)
- **Submission**: Kaggle Writeup (3 pages) + Video Demo (3 min) + Public Code

## Evaluation Criteria
| Criteria | Weight |
|----------|--------|
| Execution & Communication | 30% |
| Effective HAI-DEF Model Use | 20% |
| Product Feasibility | 20% |
| Problem Domain | 15% |
| Impact Potential | 15% |

---

## Health AI Developer Foundations (HAI-DEF) Models

### MedGemma (Multimodal Medical AI)
Gemma 3 variants trained for medical text and image comprehension.

| Model | HuggingFace ID | Params | Type |
|-------|----------------|--------|------|
| **MedGemma 1.5 4B IT** | `google/medgemma-1.5-4b-it` | 4B | Multimodal (recommended) |
| MedGemma 4B IT | `google/medgemma-4b-it` | 4B | Multimodal |
| MedGemma 4B PT | `google/medgemma-4b-pt` | 4B | Pre-trained multimodal |
| MedGemma 27B Text IT | `google/medgemma-27b-text-it` | 27B | Text-only |
| MedGemma 27B IT | `google/medgemma-27b-it` | 29B | Multimodal |

**MedGemma 1.5 Capabilities:**
- 3D Medical Imaging (CT, MRI volumes)
- 2D Medical Imaging (X-ray, dermatology, fundus, histopathology)
- Longitudinal Analysis (compare scans over time)
- Anatomical Localization (bounding boxes)
- Document Understanding (lab reports, PDFs)
- EHR Interpretation

**Benchmarks:**
| Task | Score |
|------|-------|
| MIMIC CXR (Chest X-ray) | 89.5% Macro F1 |
| MedQA | 69.1% Accuracy |
| EHRQA | 89.6% Accuracy |
| CT Classification | 61.1% Macro Accuracy |
| MRI Classification | 64.7% Macro Accuracy |

### MedSigLIP (Medical Image Encoder)
Vision encoder for classification and retrieval tasks.

| Model | HuggingFace ID | Params | Resolution |
|-------|----------------|--------|------------|
| MedSigLIP 448 | `google/medsiglip-448` | 0.9B | 448×448 |

**Use Cases:**
- Zero-shot image classification
- Semantic image retrieval
- Image-text matching
- NOT for text generation (use MedGemma instead)

### MedASR (Medical Speech Recognition)
Conformer-based ASR for medical dictation.

| Model | HuggingFace ID | Params | WER |
|-------|----------------|--------|-----|
| MedASR | `google/medasr` | 105M | 4.6% (radiology) |

**Note:** Requires `transformers >= 5.0.0`

---

## Quick Start Code

### MedGemma (Image Analysis)
```python
from transformers import pipeline
from PIL import Image
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

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

### MedSigLIP (Zero-Shot Classification)
```python
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/medsiglip-448").to("cuda")
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

texts = ["normal chest x-ray", "pneumonia", "pleural effusion"]
inputs = processor(text=texts, images=[image], padding="max_length", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits_per_image, dim=1)
```

### Loading Dataset
```python
from datasets import load_dataset

# NIH Chest X-ray (112K images, 14 pathologies)
dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", streaming=True)
```

---

## Official Tutorials (GitHub)

| Notebook | Description | Link |
|----------|-------------|------|
| Quick Start (HF) | Basic inference | [GitHub](https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_hugging_face.ipynb) |
| Quick Start (DICOM) | Medical image format | [GitHub](https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_dicom.ipynb) |
| Fine-Tuning | LoRA/QLoRA training | [GitHub](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb) |
| EHR Navigator | Agentic workflow | [GitHub](https://github.com/google-health/medgemma/blob/main/notebooks/ehr_navigator_agent.ipynb) |

---

## Datasets (HuggingFace)

### Chest X-Ray
| Dataset | ID | Size |
|---------|-----|------|
| NIH ChestX-ray14 | `alkzar90/NIH-Chest-X-ray-dataset` | 112K images |
| Pneumonia | `hf-vision/chest-xray-pneumonia` | Pediatric CXR |

### Medical QA
| Dataset | ID |
|---------|-----|
| MedQA (USMLE) | `bigbio/med_qa` |
| MedMCQA | `openlifescienceai/medmcqa` |

### 3D Imaging
| Dataset | ID | Size |
|---------|-----|------|
| CT-RATE | `ibrahimhamamci/CT-RATE` | 50K CT volumes |

---

## Development Environment

- **No local GPU** - Use Kaggle notebooks and Google Colab
- Local machine for code editing and documentation only
- Python 3.10+
- Primary compute: Kaggle (T4/P100, 30hrs/week)

### Requirements
```
transformers>=4.50.0  # For MedGemma
transformers>=5.0.0   # For MedASR (if using)
accelerate>=0.27.0
torch>=2.0.0
datasets>=2.16.0
Pillow>=10.0.0
gradio>=4.0.0
```

---

## Project Structure
```
Med Gemma/
├── CLAUDE.md                 # This file
├── README.md                 # Documentation
├── requirements.txt          # Dependencies
├── notebooks/                # Jupyter notebooks
├── src/                      # Source code
│   ├── model.py              # MedGemma wrapper
│   ├── data.py               # Data loading
│   └── agents/               # Agentic components
├── app/                      # Demo application
├── tests/                    # Tests
├── data/samples/             # Sample data
├── outputs/                  # Generated outputs
└── submission/               # Competition materials
```

---

## Important Links

| Resource | URL |
|----------|-----|
| Competition | https://www.kaggle.com/competitions/med-gemma-impact-challenge |
| HAI-DEF Docs | https://developers.google.com/health-ai-developer-foundations |
| MedGemma Model | https://huggingface.co/google/medgemma-1.5-4b-it |
| MedGemma GitHub | https://github.com/Google-Health/medgemma |
| MedSigLIP Model | https://huggingface.co/google/medsiglip-448 |
| MedASR Model | https://huggingface.co/google/medasr |
| HAI-DEF Terms | https://developers.google.com/health-ai-developer-foundations/terms |

---

## Key Notes

1. **Accept HAI-DEF terms** on Hugging Face before using models
2. **Model outputs require clinical verification** - NOT for direct patient care
3. **Fine-tuning recommended** for specific use cases (LoRA supported)
4. **MedGemma 1.5** is the latest version with 3D imaging support
5. **Use MedSigLIP** for classification tasks, MedGemma for generation
6. **MedASR** can pipe output directly to MedGemma for voice-to-diagnosis workflows
