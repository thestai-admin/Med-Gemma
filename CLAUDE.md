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

## Tech Stack
- **Model**: `google/medgemma-1.5-4b-it` (Hugging Face)
- **Dataset**: `alkzar90/NIH-Chest-X-ray-dataset` (Hugging Face)
- **Framework**: transformers >= 4.50.0, PyTorch
- **Demo**: Gradio or Streamlit
- **GPU**: Required (CUDA with bfloat16 support)

## Development Environment
- **No local GPU** - Use Kaggle notebooks and Google Colab
- Local machine for code editing and documentation only
- Python 3.10+
- Primary compute: Kaggle (T4/P100, 30hrs/week)

## Project Structure
```
Med Gemma/
├── CLAUDE.md                 # This file - project context
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── notebooks/                # Jupyter notebooks
│   ├── 01_model_exploration.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_prototype.ipynb
│   └── 04_final_solution.ipynb
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py              # MedGemma wrapper
│   ├── data.py               # Data loading
│   ├── inference.py          # Inference pipeline
│   ├── agents/               # Agentic components
│   └── utils.py              # Helpers
├── app/                      # Demo application
│   └── demo.py
├── tests/                    # Tests
├── data/samples/             # Sample data
├── outputs/                  # Generated outputs
└── submission/               # Competition submission
    ├── writeup.md
    ├── video/
    └── kaggle_notebook.ipynb
```

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo app
python app/demo.py

# Run tests
pytest tests/
```

## MedGemma Quick Reference

### Loading the Model
```python
from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

### Analyzing an Image
```python
from PIL import Image

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

### Loading Dataset
```python
from datasets import load_dataset

# NIH Chest X-ray (112K images, 14 pathologies)
dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset")
```

## Important Links
- [Competition Page](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- [MedGemma Model](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)
- [HAI-DEF Terms](https://developers.google.com/health-ai-developer-foundations/terms)
- [NIH Chest X-ray Dataset](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)

## MedGemma Capabilities
- 2D Medical Imaging: Chest X-ray, dermatology, fundus, histopathology
- 3D Medical Imaging: CT and MRI volume interpretation
- Longitudinal Analysis: Compare scans over time
- Anatomical Localization: Bounding box detection
- Document Understanding: Extract data from lab reports
- EHR Interpretation: Text-based EHR data analysis

## Notes
- HAI-DEF terms already accepted on Hugging Face
- Model outputs require clinical verification - NOT for direct patient care
- Focus on authentic clinical storytelling with Primary Care perspective
- Develop locally, test on Kaggle notebooks with GPU
