"""Repository paths, resolved from one place that does not move."""

from __future__ import annotations

from pathlib import Path

SUPPORT_DIR = Path(__file__).resolve().parent
TESTS_ROOT = SUPPORT_DIR.parent
REPO_ROOT = TESTS_ROOT.parent

DOCS_ADR = REPO_ROOT / "docs" / "adr"
PYPROJECT = REPO_ROOT / "pyproject.toml"
