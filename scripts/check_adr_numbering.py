#!/usr/bin/env python
"""Fail if two ADRs claim the same number, or a retired number returns.

This is the failure the repository actually keeps having. Six occasions are on
record: ``0002`` shipped duplicated; ``0009`` was claimed by two PRs at once and
one was renumbered to ``0010``; a third PR's ``0009`` was deleted as a collision;
``0008`` was renumbered to ``0011``, then contested between two more PRs, then
retired; and a later PR claimed ``0010`` after it was taken.

Every one was found by a human reading the tree, and each cost a review cycle
plus a renumbering pass over every reference to the ADR.

The number *is* the identifier — ``CONTEXT.md``, docstrings, other ADRs and
commit messages all cite ADRs by number alone — so a duplicate does not merely
look untidy: it makes every one of those citations ambiguous, which is how
"ADR 0002" came to mean two different documents at once.

A repo-layout check, not a test of the package, so it runs as a prek hook:

    uv run python scripts/check_adr_numbering.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = REPO_ROOT / "docs/adr"

# ``0014-response-segmentation-and-per-word-speakers.md`` -> ``0014``
ADR_FILENAME = re.compile(r"^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")

# Retired because it was contested between two PRs and its draft argued the
# opposite of what shipped. ADR 0014 records the decision; reusing the number
# would reschedule the collision that retiring it ended.
RETIRED_NUMBERS = frozenset({"0008"})

MIN_EXPECTED_ADRS = 5


def _adr_files() -> list[Path]:
    return sorted(p for p in ADR_DIR.glob("*.md") if p.name != "README.md")


def main() -> int:
    problems: list[str] = []
    files = _adr_files()

    # A check that silently examines nothing is the failure mode it guards.
    if len(files) <= MIN_EXPECTED_ADRS:
        problems.append(
            f"expected a populated {ADR_DIR.relative_to(REPO_ROOT)}, found {len(files)} file(s) — "
            "this check cannot be trusted until that is explained"
        )

    malformed = [p.name for p in files if not ADR_FILENAME.match(p.name)]
    if malformed:
        problems.append(
            "ADR filenames must be NNNN-lower-kebab-case.md so the number can be read "
            f"back: {', '.join(malformed)}"
        )

    by_number: dict[str, list[str]] = defaultdict(list)
    for path in files:
        match = ADR_FILENAME.match(path.name)
        if match:
            by_number[match.group(1)].append(path.name)

    for number, names in sorted(by_number.items()):
        if len(names) > 1:
            problems.append(
                f"ADR {number} is claimed by {len(names)} files, so every reference to it is "
                f"ambiguous — renumber the one that merged later and repoint its references: "
                f"{', '.join(sorted(names))}"
            )

    reused = sorted(p.name for p in files if p.name[:4] in RETIRED_NUMBERS)
    if reused:
        problems.append(f"these ADR numbers are retired and must not return: {', '.join(reused)}")

    if problems:
        print("ADR numbering check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
