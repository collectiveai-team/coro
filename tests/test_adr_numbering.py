"""ADR numbers identify an ADR, so two files may never claim the same one.

This is the failure the repository actually keeps having. Six occasions are on
record: ``0002`` shipped duplicated; ``0009`` was claimed by two PRs at once and
one was renumbered to ``0010``; a third PR's ``0009`` was deleted as a collision;
``0008`` was renumbered to ``0011``, then contested between two more PRs, then
retired; and a later PR claimed ``0010`` after it was taken.

Every one was found by a human reading the tree, and each cost a review cycle
plus a renumbering pass over every reference to the ADR. Nothing checked it.

The number is the identifier — ``CONTEXT.md``, docstrings, other ADRs and commit
messages all cite ADRs by number alone — so a duplicate does not merely look
untidy: it makes every one of those citations ambiguous, which is how "ADR 0002"
came to mean two different documents in the same repository.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = REPO_ROOT / "docs/adr"

# ``0014-response-segmentation-and-per-word-speakers.md`` -> ``0014``
ADR_FILENAME = re.compile(r"^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")


def _adr_files() -> list[Path]:
    return sorted(p for p in ADR_DIR.glob("*.md") if p.name != "README.md")


def test_adr_directory_is_not_empty() -> None:
    """A check that silently examines nothing is the failure mode it guards."""
    assert len(_adr_files()) > 5, f"expected a populated ADR directory, found {_adr_files()}"


def test_every_adr_filename_carries_a_four_digit_number() -> None:
    """The number has to be parseable for uniqueness to mean anything."""
    malformed = [p.name for p in _adr_files() if not ADR_FILENAME.match(p.name)]
    assert not malformed, (
        "ADR filenames must be NNNN-lower-kebab-case.md so the number can be read "
        f"back: {malformed}"
    )


def test_no_two_adrs_claim_the_same_number() -> None:
    by_number: dict[str, list[str]] = defaultdict(list)
    for path in _adr_files():
        match = ADR_FILENAME.match(path.name)
        if match:
            by_number[match.group(1)].append(path.name)

    duplicates = {number: names for number, names in by_number.items() if len(names) > 1}
    assert not duplicates, (
        "Two ADRs claim one number, so every reference to it is ambiguous. Renumber "
        "the one that merged later and repoint its references:\n"
        + "\n".join(f"  {number}: {', '.join(sorted(names))}" for number, names in duplicates.items())
    )


def test_retired_numbers_are_not_reused() -> None:
    """``0008`` was contested, then retired; a file must not claim it again.

    Its content is recorded by ADR 0014. Reusing the number would reschedule the
    collision it was retired to end. Kept here rather than as a prose needle
    because this is a check about the tree, not about vocabulary.
    """
    retired = {"0008"}
    reused = sorted(p.name for p in _adr_files() if p.name[:4] in retired)
    assert not reused, f"these ADR numbers are retired and must not return: {reused}"
