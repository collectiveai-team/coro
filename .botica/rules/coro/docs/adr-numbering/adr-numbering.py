# /// script
# requires-python = ">=3.10"
# dependencies = ["botica[sdk]==0.18.0"]
#
# [tool.uv]
# # The exact pin is the botica that prek.toml and CI already run; the 7-day cooldown of
# # pyproject.toml would otherwise hold it back for a week after each upgrade.
# exclude-newer-package = { botica = false }
# ///
"""Fail if two ADRs claim the same number, or a retired number returns.

This is the failure the repository actually keeps having. Six occasions are on
record: ``0002`` shipped duplicated; ``0009`` was claimed by two PRs at once and
one was renumbered to ``0010``; a third PR's ``0009`` was deleted as a collision;
``0008`` was renumbered to ``0011``, then contested between two more PRs, then
retired; and a later PR claimed ``0010`` after it was taken.

The number *is* the identifier: ``CONTEXT.md``, docstrings, other ADRs and commit
messages all cite ADRs by number alone, so a duplicate makes every one of those
citations ambiguous, which is how "ADR 0002" came to mean two different documents
at once.

The rule reads ``docs/adr`` itself and reports each problem on the ADR file that
causes it, so ``botica lint --staged`` fails on the file a commit adds.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from botica.sdk import rule, serve

RULE_ID = "coro/adr-numbering"

ADR_DIR = "docs/adr"

# ``0014-response-segmentation-and-per-word-speakers.md`` -> ``0014``
ADR_FILENAME = re.compile(r"^(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")

# Retired because it was contested between two PRs and its draft argued the
# opposite of what shipped. ADR 0014 records the decision; reusing the number
# would reschedule the collision that retiring it ended.
RETIRED_NUMBERS = frozenset({"0008"})

MIN_EXPECTED_ADRS = 5

Finding = tuple[str, int, int, str]


def _adr_files(root: Path) -> list[Path]:
    return sorted(p for p in (root / ADR_DIR).glob("*.md") if p.name != "README.md")


def _first_line(path: Path) -> tuple[int, int]:
    """Return the byte range of the first line of `path`, where a finding about it sits."""
    try:
        data = path.read_bytes()
    except OSError:
        return 0, 0
    end = data.find(b"\n")
    return 0, len(data) if end == -1 else end


def _at(root: Path, path: Path, message: str) -> Finding:
    start, end = _first_line(path)
    return path.relative_to(root).as_posix(), start, end, message


def _problems(root: Path, files: list[Path]) -> Iterator[Finding]:
    # A check that silently examines nothing is the failure mode it guards.
    if len(files) <= MIN_EXPECTED_ADRS:
        anchor = root / ADR_DIR / "README.md"
        if not anchor.is_file() and files:
            anchor = files[0]
        yield (
            anchor.relative_to(root).as_posix(),
            0,
            0,
            f"expected a populated {ADR_DIR}, found {len(files)} file(s) — "
            "this check cannot be trusted until that is explained",
        )

    malformed = [p for p in files if not ADR_FILENAME.match(p.name)]
    for path in malformed:
        yield _at(
            root,
            path,
            "ADR filenames must be NNNN-lower-kebab-case.md so the number can be read "
            f"back: {', '.join(p.name for p in malformed)}",
        )

    by_number: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        match = ADR_FILENAME.match(path.name)
        if match:
            by_number[match.group(1)].append(path)

    for number, paths in sorted(by_number.items()):
        if len(paths) > 1:
            names = ", ".join(sorted(p.name for p in paths))
            for path in paths:
                yield _at(
                    root,
                    path,
                    f"ADR {number} is claimed by {len(paths)} files, so every reference to it "
                    "is ambiguous — renumber the one that merged later and repoint its "
                    f"references: {names}",
                )

    reused = [p for p in files if p.name[:4] in RETIRED_NUMBERS]
    for path in reused:
        yield _at(
            root,
            path,
            "these ADR numbers are retired and must not return: "
            f"{', '.join(sorted(p.name for p in reused))}",
        )


@rule(RULE_ID, ["*"], severity="error", scope="project")
def check(root: str, paths: list[str], options: dict[str, object]) -> Iterator[Finding]:
    """Report the duplicated, malformed and retired ADR numbers under ``docs/adr``."""
    base = Path(root)
    yield from _problems(base, _adr_files(base))


if __name__ == "__main__":
    serve()
