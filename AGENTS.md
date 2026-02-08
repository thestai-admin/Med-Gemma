# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains core application code.
- `src/agents/` contains domain agents (`intake.py`, `imaging.py`, `reasoning.py`, `guidelines.py`, `education.py`) and `orchestrator.py` as the main integration point.
- `src/edge/` contains the Edge AI module: `inference.py` (EdgeClassifier), `quantize.py` (ONNX export + INT8), `benchmark.py` (latency/accuracy benchmarks).
- `src/eval/` contains deterministic evaluation utilities for CXR binary classification.
- `app/demo.py` is the Gradio entrypoint for local interactive testing (7 tabs including Patient Education).
- `tests/` contains unit and integration-style tests with shared fixtures in `tests/conftest.py`.
- `data/guidelines/chunks.json` stores RAG guideline chunks; generated artifacts should go under `data/` or `outputs/`.
- `notebooks/` contains Kaggle/Colab experimentation and submission notebooks.
- `scripts/` contains utility scripts: `prepare_guidelines.py`, `export_edge_model.py`, `run_edge_benchmark.py`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime and test dependencies.
- `python app/demo.py` runs the Gradio demo locally (GPU recommended).
- `python scripts/prepare_guidelines.py` generates guideline embeddings for the RAG agent.
- `python scripts/export_edge_model.py` exports MedSigLIP to ONNX and quantizes to INT8.
- `python scripts/run_edge_benchmark.py` benchmarks the edge classifier on CPU.
- `pytest` runs the full test suite (42 tests, all mock-based, no GPU).
- `pytest -k education` runs education agent tests.
- `pytest -k edge` runs edge module tests.
- `pytest -k longitudinal` runs targeted tests by keyword during feature work.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and type hints where practical.
- Use `snake_case` for functions/modules, `PascalCase` for classes, and clear clinical/domain names.
- Keep dataclasses and typed result objects explicit; prefer small, composable methods.
- Match existing docstring style (triple-quoted, concise purpose + behavior).

## Testing Guidelines
- Framework: `pytest` with fixtures in `tests/conftest.py`.
- Name tests as `tests/test_<feature>.py` and test functions as `test_<behavior>()`.
- Use the `requires_gpu` marker configured in `conftest.py` for GPU-dependent tests.
- Add or update tests for any behavior change in agents or orchestrator flows.

## Commit & Pull Request Guidelines
- Match repository history: imperative, sentence-style commit subjects.
- Keep commits focused on one logical change and include tests/docs updates when relevant.

## Security & Configuration Tips
- Do not commit secrets. Set `HF_TOKEN` through environment variables, Kaggle Secrets, or Colab secrets.
- For notebook runs, set `TORCHDYNAMO_DISABLE=1` before importing `torch` to avoid runtime issues in Kaggle environments.
