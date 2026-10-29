# /// script
# requires-python = ">=3.10"
# dependencies = ["botica[sdk]==0.18.0"]
#
# [tool.uv]
# # The exact pin is the botica that prek.toml and CI already run; the 7-day cooldown of
# # pyproject.toml would otherwise hold it back for a week after each upgrade.
# exclude-newer-package = { botica = false }
# ///
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
  evidence that rejected it, which is how a settled question quietly becomes
  arguable again.

A call site can appear in any module, so the rule reads the git index (tracked and
staged files) rather than the files botica sends it, which under ``--staged`` or
``--diff`` are only the changed ones. Outside a git work tree (the fixtures of the
rule) it reads the files botica sends.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from botica.sdk import rule, serve

RULE_ID = "coro/flicker-rule-rejected"

SYMBOL = "realign_speaker_flicker"

# The module that defines it, the tests that measure it and the ADR that records why
# it is unused. The folder of this rule (its script, README and fixtures) is allowed too.
ALLOWED_REFERENCES = frozenset(
    {
        "coro/core/realignment.py",
        "tests/core/test_core_realignment.py",
        "docs/adr/0014-response-segmentation-and-per-word-speakers.md",
    }
)

MUST_EXIST = (
    "coro/core/realignment.py",
    "tests/core/test_core_realignment.py",
)

SCANNED_SUFFIXES = frozenset({".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".cfg", ".txt"})

MIN_EXPECTED_FILES = 100

Finding = tuple[str, int, int, str]


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(root), *args],  # noqa: S607
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _candidate_files(root: Path, sent: list[str]) -> list[str]:
    """Every tracked or staged file that could hold a reference.

    The git index excludes untracked scratch output, virtualenvs and build artifacts
    by construction, and includes *staged* additions, so a brand-new caller is caught
    at commit time because ``git add`` put it there.
    """
    toplevel = _git(root, "rev-parse", "--show-toplevel")
    listing = None
    if toplevel and Path(toplevel.strip()).resolve() == root.resolve():
        listing = _git(root, "ls-files", "-z")
    names = listing.split("\0") if listing is not None else sent
    return [n for n in names if n and PurePosixPath(n).suffix in SCANNED_SUFFIXES]


def _own_folder(root: Path) -> str | None:
    folder = Path(__file__).resolve().parent
    try:
        return folder.relative_to(root.resolve()).as_posix() + "/"
    except ValueError:
        return None


def _references(root: Path, name: str) -> Iterator[tuple[int, int]]:
    """Yield the byte ranges of `SYMBOL` in the file `name`."""
    try:
        data = (root / name).read_bytes()
    except OSError:
        return
    symbol = SYMBOL.encode()
    start = data.find(symbol)
    while start != -1:
        yield start, start + len(symbol)
        start = data.find(symbol, start + len(symbol))


@rule(RULE_ID, ["*"], severity="error", scope="project")
def check(root: str, paths: list[str], options: dict[str, object]) -> Iterator[Finding]:
    """Report each reference to the rejected rule, and its deleted evidence."""
    base = Path(root)
    files = _candidate_files(base, paths)
    own = _own_folder(base)

    # A check that silently scans nothing is the failure mode it guards.
    minimum = options.get("min-files", MIN_EXPECTED_FILES)
    if isinstance(minimum, int) and len(files) < minimum:
        yield (
            files[0] if files else ".",
            0,
            0,
            f"expected a populated repo, scanned {len(files)} files — this check "
            "cannot be trusted until that is explained",
        )

    for name in files:
        if name in ALLOWED_REFERENCES or (own and name.startswith(own)):
            continue
        for start, end in _references(base, name):
            yield (
                name,
                start,
                end,
                f"{SYMBOL} must not be reachable from the default assembly path. ADR 0014 "
                "dropped the sandwich rule after measuring it; the module survives only as "
                "the record of that measurement (issue 17).",
            )

    for path in MUST_EXIST:
        if not (base / path).is_file():
            yield (
                path,
                0,
                0,
                "the rejection evidence must not be deleted — losing it is how the question "
                f"gets re-litigated from scratch. Missing: {path}",
            )


if __name__ == "__main__":
    serve()
