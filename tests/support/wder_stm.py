"""Reference/hypothesis STM pairs for the WDER tests."""

from __future__ import annotations

from pathlib import Path

type Line = tuple[str, float, float, str]
"""One STM segment: speaker, start, end and text."""

REF: list[Line] = [
    ("A", 0.0, 1.0, "alpha"),
    ("B", 1.0, 2.0, "bravo"),
    ("A", 2.0, 3.0, "charlie"),
    ("B", 3.0, 4.0, "delta"),
]
"""Reference used by most cases: two speakers alternating word by word."""


def write_stm(path: Path, lines: list[Line]) -> Path:
    """Write an STM file of session ``sess`` from (speaker, start, end, text) tuples."""
    path.write_text(
        "\n".join(f"sess 1 {spk} {start:.3f} {end:.3f} {text}" for spk, start, end, text in lines)
        + "\n"
    )
    return path
