#!/usr/bin/env python3
"""
Competition submission readiness checker.

Runs lightweight file/content checks so competition packaging issues are caught
before final submission.

Usage:
    python3 scripts/check_submission_readiness.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _exists(path: Path, name: str) -> CheckResult:
    return CheckResult(
        name=name,
        ok=path.exists(),
        detail=f"{path} {'found' if path.exists() else 'missing'}",
    )


def _scan_placeholders(paths: Iterable[Path], patterns: Iterable[str]) -> List[CheckResult]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    findings: List[CheckResult] = []
    for path in paths:
        if not path.exists():
            findings.append(CheckResult(
                name=f"placeholder_scan:{path}",
                ok=False,
                detail=f"{path} missing",
            ))
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = []
        for pat in compiled:
            for match in pat.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"line {line}: /{pat.pattern}/")
        findings.append(CheckResult(
            name=f"placeholder_scan:{path}",
            ok=not hits,
            detail=f"{len(hits)} placeholders" if hits else "none found",
        ))
    return findings


def _check_notebook_outputs(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name=f"notebook_outputs:{path}",
            ok=False,
            detail=f"{path} missing",
        )

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult(
            name=f"notebook_outputs:{path}",
            ok=False,
            detail=f"invalid JSON: {exc}",
        )

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    with_outputs = sum(1 for c in code_cells if c.get("outputs"))

    return CheckResult(
        name=f"notebook_outputs:{path}",
        ok=with_outputs > 0,
        detail=f"{with_outputs} code cells with outputs",
    )


def main() -> int:
    checks: List[CheckResult] = []

    # Core docs and submission artifacts
    checks.extend([
        _exists(ROOT / "README.md", "readme"),
        _exists(ROOT / "submission" / "writeup.md", "writeup"),
        _exists(ROOT / "submission" / "kaggle_writeup.md", "kaggle_writeup"),
        _exists(ROOT / "submission" / "video" / "video_script.md", "video_script"),
    ])

    # RAG assets
    checks.extend([
        _exists(ROOT / "data" / "guidelines" / "chunks.json", "guideline_chunks"),
        _exists(ROOT / "data" / "guidelines" / "embeddings.npz", "guideline_embeddings"),
    ])

    # Edge assets
    checks.extend([
        _exists(ROOT / "models" / "edge" / "medsiglip_fp32.onnx", "edge_fp32_onnx"),
        _exists(ROOT / "models" / "edge" / "medsiglip_int8.onnx", "edge_int8_onnx"),
        _exists(ROOT / "models" / "edge" / "text_embeddings.npy", "edge_text_embeddings"),
        _exists(ROOT / "models" / "edge" / "preprocess_config.json", "edge_preprocess_config"),
    ])

    # Primary notebook should include outputs for submission evidence.
    checks.append(_check_notebook_outputs(ROOT / "notebooks" / "primacare-ai-submission.ipynb"))

    # Placeholder scan in user-facing docs.
    placeholder_paths = [
        ROOT / "README.md",
        ROOT / "submission" / "writeup.md",
        ROOT / "submission" / "kaggle_writeup.md",
        ROOT / "submission" / "video" / "video_script.md",
    ]
    placeholder_patterns = [
        r"YOUR_USERNAME",
        r"add your video link",
        r"update with your Kaggle notebook URL",
        r"\bTODO\b",
        r"\bTBD\b",
    ]
    checks.extend(_scan_placeholders(placeholder_paths, placeholder_patterns))

    passed = [c for c in checks if c.ok]
    failed = [c for c in checks if not c.ok]

    print("=" * 72)
    print("MedGemma Competition Readiness Check")
    print("=" * 72)
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"[{status}] {c.name:<40} {c.detail}")

    print("-" * 72)
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print("=" * 72)

    if failed:
        print("Readiness check failed. Resolve FAIL items before final submission.")
        return 1

    print("Readiness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
