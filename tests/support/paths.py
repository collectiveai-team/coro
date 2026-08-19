"""Repository paths, resolved from one place that does not move.

Tests that read a tracked file (an ADR, `pyproject.toml`) previously counted
`..` from their own location, which silently repoints whenever a module moves
between directories of the mirrored test tree.
"""

from __future__ import annotations

from pathlib import Path

# tests/support/paths.py -> tests/support -> tests -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ADR = REPO_ROOT / "docs" / "adr"
PYPROJECT = REPO_ROOT / "pyproject.toml"
