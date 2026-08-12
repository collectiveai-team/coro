#!/usr/bin/env python
"""Fail if the rejected flicker rule becomes reachable, or its evidence is deleted.

``coro/core/realignment.py`` implements the "sandwich rule": a word whose
neighbours agree on a speaker gets relabelled to match them. It was measured on
the 6-clip WDER workload, it did not pay, and ADR 0014 dropped it from response
assembly. The module was kept deliberately, as the record of that measurement
(issue 17).

Two checks, and the pairing is the point:

- nothing outside the module, its own tests and the ADR may reference
  ``realign_speaker_flicker``, so the rule cannot creep back into assembly;
- the module and its tests must continue to exist, so nobody "cleans up" the
  evidence that rejected it — which is how a settled question quietly becomes
  arguable again.

This is an invariant about which code may call what. A call site can appear in
any module, so it is checked by reading the tracked tree rather than by importing
anything — a repo-layout check, not a test of the package:

    uv run python scripts/check_flicker_rule_rejected.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SYMBOL = "realign_speaker_flicker"

# The module that defines it, the tests that measure it, the ADR that records why
# it is unused, and this script.
ALLOWED_REFERENCES = frozenset(
    {
        "coro/core/realignment.py",
        "tests/test_core_realignment.py",
        "docs/adr/0014-response-segmentation-and-per-word-speakers.md",
        "scripts/check_flicker_rule_rejected.py",
    }
)

MUST_EXIST = (
    "coro/core/realignment.py",
    "tests/test_core_realignment.py",
)

SCANNED_SUFFIXES = frozenset({".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".cfg", ".txt"})

MIN_EXPECTED_FILES = 100


def _tracked_text_files() -> list[str]:
    """Every tracked file that could hold a reference.

    Reads the git index rather than walking the tree so untracked scratch output,
    virtualenvs and build artifacts are excluded by construction.

    The index includes *staged* additions, which is what makes this work as a
    commit hook: a brand-new caller is caught because ``git add`` put it there.
    A file that is neither tracked nor staged is invisible here — correctly, since
    it is not part of the commit and not part of a CI checkout.
    """
    listing = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [name for name in listing.split("\0") if name and Path(name).suffix in SCANNED_SUFFIXES]


def main() -> int:
    problems: list[str] = []

    tracked = _tracked_text_files()
    # A check that silently scans nothing is the failure mode it guards.
    if len(tracked) < MIN_EXPECTED_FILES:
        problems.append(
            f"expected a populated repo, scanned {len(tracked)} files — this check "
            "cannot be trusted until that is explained"
        )

    offenders: list[str] = []
    for name in tracked:
        if name in ALLOWED_REFERENCES:
            continue
        text = (REPO_ROOT / name).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if SYMBOL in line:
                offenders.append(f"{name}:{lineno}: {line.strip()[:120]}")

    if offenders:
        problems.append(
            f"{SYMBOL} must not be reachable from the default assembly path. ADR 0014 "
            "dropped the sandwich rule after measuring it; the module survives only as "
            "the record of that measurement (issue 17). Found at:\n"
            + "\n".join(f"      {o}" for o in offenders)
        )

    missing = [path for path in MUST_EXIST if not (REPO_ROOT / path).is_file()]
    if missing:
        problems.append(
            "the rejection evidence must not be deleted — losing it is how the question "
            f"gets re-litigated from scratch. Missing: {', '.join(missing)}"
        )

    if problems:
        print("Flicker-rule check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
