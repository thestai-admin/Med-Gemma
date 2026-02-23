# Repository Guidelines for MedGemma

Guidelines for AI agents working on this medical AI diagnostic support system.

## Project Structure

```
MedGemma/
├── src/
│   ├── agents/           # Domain agents (intake, imaging, reasoning, guidelines, education, orchestrator)
│   ├── edge/             # ONNX-based CPU classifier, quantization, benchmarks
│   ├── eval/             # CXR binary classification metrics
│   └── model.py          # MedGemma model wrapper
├── app/demo.py           # Gradio demo (7 tabs)
├── tests/                # 42 mock-based tests (no GPU required)
├── scripts/              # prepare_guidelines.py, export_edge_model.py, run_edge_benchmark.py
├── data/guidelines/     # RAG guideline chunks
└── notebooks/            # Kaggle/Colab experimentation
```

## Build, Test, and Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app/demo.py  # GPU recommended
```

### Running Tests
```bash
pytest                              # Full test suite (42 tests)
pytest tests/test_education.py     # Single test file
pytest tests/test_education.py::test_educate_basic_level_returns_patient_education  # Single test
pytest -k education                # Tests matching keyword
pytest -v                          # Verbose output
pytest --cov=src --cov-report=html # With coverage
```

### Scripts
```bash
python scripts/prepare_guidelines.py   # Generate guideline embeddings
python scripts/export_edge_model.py    # Export to ONNX + INT8
python scripts/run_edge_benchmark.py    # CPU inference benchmark
```

### Linting (Optional - Not Configured)
```bash
pip install ruff mypy
ruff check src/
ruff format src/
mypy src/
```

## Code Style Guidelines

### General Principles
- **PEP 8 with 4-space indentation** (not tabs)
- Use type hints where practical
- Keep functions small and focused (single responsibility)
- Prefer explicit over implicit

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Modules/Functions | `snake_case` | `intake_agent`, `create_patient_context` |
| Classes | `PascalCase` | `PatientEducationAgent`, `PrimaCareResult` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_TOKENS`, `DEFAULT_THRESHOLD` |
| Private members | Leading underscore | `_model`, `_intake_agent` |

### Import Organization
Group in order (separate with blank lines):
1. Standard library
2. Third-party packages
3. Local application imports

```python
# Standard library
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

# Third-party
from PIL import Image
import numpy as np

# Local application
from .intake import IntakeAgent, PatientContext
```

### Type Hints
Use for function parameters, return types, and complex data structures.

```python
def run(
    chief_complaint: str,
    history: str = "",
    xray_image: Optional[Union[Image.Image, str, Path]] = None,
    age: Optional[int] = None,
) -> PrimaCareResult:
    ...
```

### Docstrings
Use triple-quoted docstrings with concise purpose + behavior:

```python
def educate(self, result, reading_level: str = "basic") -> PatientEducation:
    """
    Generate patient-friendly education from a clinical result.

    Args:
        result: PrimaCareResult with clinical assessment
        reading_level: One of "basic", "intermediate", "detailed"

    Returns:
        PatientEducation with simplified content and glossary
    """
```

### Dataclasses
Use `@dataclass` for structured data containers:

```python
@dataclass
class PatientEducation:
    """Patient-friendly education output from a clinical report."""
    reading_level: str
    simplified_diagnosis: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)
```

### Error Handling
- Catch specific exceptions, not bare `except:`
- Log/print warnings for recoverable failures
- Let unexpected exceptions propagate in non-user-facing code

```python
try:
    guidelines_result = self.guidelines_agent.get_recommendations(...)
    result.guidelines_result = guidelines_result
except Exception as e:
    print(f"Warning: Guidelines retrieval failed: {e}")
    result.processing_steps.append("guidelines_failed")
```

### Property-Based Lazy Loading
```python
@property
def model(self):
    """Lazy load MedGemma model."""
    if self._model is None:
        from ..model import MedGemma
        self._model = MedGemma()
    return self._model
```

### Boolean Parameters
Use descriptive names: `include_classification: bool = True` (not `run: bool = True`)

### Testing Conventions
- File: `tests/test_<feature>.py`
- Function: `test_<behavior>()` (lowercase with underscores)
- Use fixtures from `tests/conftest.py`
- Mark GPU tests with `@pytest.mark.requires_gpu`

```python
def test_educate_basic_level_returns_patient_education(education_agent, mock_result):
    """Test that educate() returns a populated PatientEducation at basic level."""
    education = education_agent.educate(mock_result, reading_level="basic")
    assert isinstance(education, PatientEducation)
    assert education.reading_level == "basic"
```

### Configuration and Secrets
- Never commit secrets
- Use environment variables: `HF_TOKEN`, `KAGGLE_KEY`, etc.
- Store in `.env` files (add to `.gitignore`)

### Performance Considerations
- Use `profile=True` for per-stage timings
- Use `ThreadPoolExecutor` for parallel execution
- Consider T4 GPU memory constraints

### File Organization
- One class per file for public classes
- Use `__all__` to declare public API
- Generated artifacts go in `data/` or `outputs/`

## Additional Tips

- For Kaggle/Colab: set `TORCHDYNAMO_DISABLE=1` before importing torch
- Use `fast_mode=True` for lower-latency (disables guidelines, binary classification only)
