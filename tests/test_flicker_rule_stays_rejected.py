"""The punctuation-majority word-relabelling rule stays out of the default path.

``coro/core/realignment.py`` implements the "sandwich rule": a word whose
neighbours agree on a speaker gets relabelled to match them. It was measured on
the 6-clip WDER workload, it did not pay, and ADR 0014 dropped it from response
assembly. The module was kept deliberately, as the record of that measurement
(issue 17).

Two assertions, and the pairing is the point:

- nothing outside the module, its own tests and the ADR may call
  ``realign_speaker_flicker``, so the rule cannot creep back into assembly;
- the module and its tests must continue to exist, so nobody "cleans up" the
  evidence that rejected it — which is how a settled question quietly becomes
  arguable again.

This is an invariant about which code may call what, checked by reading the
tracked tree because a call site can appear in any module. It is not a rule about
vocabulary: the name is banned as a *caller*, not as a word.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SYMBOL = "realign_speaker_flicker"

# The module that defines it, the tests that measure it, the ADR that records
# why it is not used, and this file.
ALLOWED_CALLERS = (
    "coro/core/realignment.py",
    "tests/test_core_realignment.py",
    "docs/adr/0014-response-segmentation-and-per-word-speakers.md",
    "tests/test_flicker_rule_stays_rejected.py",
)

SCANNED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".cfg", ".txt"}


def _tracked_text_files() -> list[Path]:
    """Every tracked file that could hold a call site.

    Uses the git index rather than a directory walk so untracked scratch output,
    virtualenvs and build artifacts are excluded by construction.
    """
    listing = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [
        REPO_ROOT / name
        for name in listing.split("\0")
        if name and Path(name).suffix in SCANNED_SUFFIXES
    ]


@pytest.fixture(scope="module")
def tracked_files() -> list[Path]:
    files = _tracked_text_files()
    # A check that silently scans nothing is the exact failure mode it exists to
    # prevent, so assert the corpus is real before trusting a pass below.
    assert len(files) > 100, f"expected a populated repo, scanned {len(files)} files"
    return files


def test_flicker_correction_is_not_reachable_from_assembly(tracked_files: list[Path]) -> None:
    offenders: list[str] = []
    for path in tracked_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_CALLERS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if SYMBOL in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")

    assert not offenders, (
        f"{SYMBOL} must not be reachable from the default assembly path. ADR 0014 "
        "dropped the sandwich rule after measuring it; the module survives only as "
        "the record of that measurement (issue 17).\n\nFound at:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_the_rejection_evidence_still_exists() -> None:
    """Paired with the test above on purpose.

    One stops the rule coming back; this one stops someone deleting the
    measurement that rejected it. Losing the evidence is how the question gets
    re-litigated from scratch.
    """
    assert (REPO_ROOT / "coro/core/realignment.py").is_file()
    assert (REPO_ROOT / "tests/test_core_realignment.py").is_file()
