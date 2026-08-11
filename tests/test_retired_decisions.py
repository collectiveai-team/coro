"""Guard the repository against decisions that were made and then un-made.

A decision gets recorded in an ADR, the code changes, and prose somewhere else
keeps asserting the superseded version — a retired ADR number, a renamed
function, a guarantee the response no longer offers. Each incident cost a review
cycle to rediscover, and one shipped a docstring that contradicted its own ADR in
the same commit that wrote the ADR.

Each entry below is a string with **no legitimate present-tense use** in this
repository. ADRs that must describe what they superseded name themselves in
``allow``; adding any other exception means editing ``allow`` with a reason,
which is a visible diff in review.

**Scope, and what this cannot do.** A needle belongs here only when the thing it
names no longer exists or is no longer true — a *fact* about the tree, not a
preference about its direction. Two limits follow, and both have been hit:

- It is not a place to make a decision permanent. A judgement taken in a context
  can be revisited; banning its vocabulary enforces a finality the decision never
  had, and blocks the very thing an ADR exists for, which is recording what was
  considered and why it was not chosen. One needle was removed for exactly this.
- It only catches strings someone already thought to list, so it always describes
  the *last* incident and never the next one. A stubbed confidence documented as
  a measured one was the same species of defect and this guard was blind to it,
  because no retired name was involved.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

ADR_0014 = "docs/adr/0014-response-segmentation-and-per-word-speakers.md"
SELF = "tests/test_retired_decisions.py"

# Only text a human reads or a machine executes. Lock files and binaries carry
# strings this guard has no opinion about.
SCANNED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".cfg", ".txt"}


@dataclass(frozen=True)
class Retired:
    """One retired decision, and where it is still allowed to be named."""

    needle: str
    reason: str
    allow: tuple[str, ...] = field(default=())


RETIRED = (
    Retired(
        needle="ADR 0008",
        reason=(
            "ADR 0008 is retired: it was contested between two PRs and its draft argued "
            "the opposite of what shipped. ADR 0014 records the decision. Never reuse 0008."
        ),
        allow=(ADR_0014, SELF),
    ),
    Retired(
        needle="0008-word-level",
        reason="The ADR 0008 file was replaced by ADR 0014; the filename must not return.",
        allow=(ADR_0014, SELF),
    ),
    Retired(
        needle="build_response_segments",
        reason=(
            "ADR 0014 replaced the plural builder with build_response_segment, which returns "
            "one ResponseSegment or None because a run is no longer split by speaker."
        ),
        allow=(SELF,),
    ),
    Retired(
        needle="TranscriptSegment",
        reason=(
            "Removed by ADR 0014. It existed only as the intermediate the old "
            "'group, then stamp a speaker' builder needed."
        ),
        allow=(ADR_0014, SELF),
    ),
    Retired(
        needle="speaker-homogeneous",
        reason=(
            "segments[].speaker is a duration-weighted majority, not a homogeneity "
            "guarantee (ADR 0014): a segment may span a speaker turn."
        ),
        allow=(ADR_0014, SELF),
    ),
)
# A third vendor dialect was *not* retired, and no needle guards it. ADR 0015
# declines one specific vendor because its asynchronous contract needs job-state
# infrastructure coro does not have — a cost judgement in a context, not a closed
# door and not a cap on how many dialects may exist. Banning the name here was a
# category error: it enforced a permanence the decision never had, and it
# forbade the one thing an ADR is for, which is writing down what was considered
# and why it was not chosen.

# The sandwich rule was measured, rejected, and kept as evidence (issue 17). The
# module and its tests must survive; the default assembly path must not call it.
FLICKER_CALLER_ALLOW = (
    "coro/core/realignment.py",
    "tests/test_core_realignment.py",
    ADR_0014,
    SELF,
)


def _tracked_text_files() -> list[Path]:
    """Every tracked file this guard has an opinion about.

    Uses the git index rather than a directory walk so untracked scratch output,
    virtualenvs and build artifacts are excluded by construction.
    """
    listing = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
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
    # A guard that silently scans nothing is the exact failure mode it exists to
    # prevent, so assert the corpus is real before trusting any pass below.
    assert len(files) > 100, f"expected a populated repo, scanned {len(files)} files"
    return files


@pytest.mark.parametrize("retired", RETIRED, ids=lambda r: r.needle)
def test_retired_decision_is_not_reintroduced(retired: Retired, tracked_files: list[Path]) -> None:
    offenders: list[str] = []
    for path in tracked_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in retired.allow:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if retired.needle.lower() in line.lower():
                offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")

    assert not offenders, (
        f"'{retired.needle}' is retired.\n{retired.reason}\n\nFound at:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_flicker_correction_stays_out_of_the_default_path(tracked_files: list[Path]) -> None:
    """The module is kept as measured evidence; nothing may call it in assembly."""
    offenders: list[str] = []
    for path in tracked_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in FLICKER_CALLER_ALLOW:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "realign_speaker_flicker" in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")

    assert not offenders, (
        "realign_speaker_flicker must not be reachable from the default assembly "
        "path. ADR 0014 dropped the sandwich rule; the module survives only as the "
        "record of the measurement that rejected it (issue 17).\n\nFound at:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_realignment_module_still_exists() -> None:
    """The rejection evidence must not be deleted either.

    Paired with the guard above on purpose: one test stops the rule coming back,
    this one stops someone 'cleaning up' the measurement that rejected it, which
    is how a settled question becomes arguable again.
    """
    assert (REPO_ROOT / "coro/core/realignment.py").is_file()
    assert (REPO_ROOT / "tests/test_core_realignment.py").is_file()
