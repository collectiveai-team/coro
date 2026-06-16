"""Side-by-side reference/hypothesis alignment visualizations.

Thin convenience over meeteval's own ``meeteval-viz html`` tool: discover the
per-session (ref, hyp) STM pairs a quality run writes, combine them, and invoke
meeteval-viz to render the standard word-aligned side-by-side view (plus the
synced ``side_by_side_sync.html`` when multiple alignment algorithms are given).

The actual alignment/rendering is entirely meeteval's; this module only adds
bench-out-dir discovery so a whole run can be visualised in one call.
meeteval (and simplejson) ship in the optional ``bench`` extra.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def discover_quality_pairs(out_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find (session_id, ref_stm, hyp_stm) triples in a quality run out-dir.

    Pairs ``hyp/<id>.hyp.stm`` with ``ref/<id>.ref.stm``; sessions missing
    either file are skipped. Sorted by session id.
    """
    ref_dir = out_dir / "ref"
    hyp_dir = out_dir / "hyp"
    if not hyp_dir.is_dir():
        return []
    pairs: list[tuple[str, Path, Path]] = []
    for hyp in sorted(hyp_dir.glob("*.hyp.stm")):
        session_id = hyp.name[: -len(".hyp.stm")]
        ref = ref_dir / f"{session_id}.ref.stm"
        if ref.exists():
            pairs.append((session_id, ref, hyp))
    return pairs


def combine_session_stms(
    pairs: list[tuple[str, Path, Path]],
    dest_dir: Path,
) -> tuple[Path, Path] | None:
    """Concatenate per-session ref/hyp STMs into single multi-session STM files.

    meeteval keys sessions by the STM session-id column, so concatenating the
    per-session files yields one corpus that meeteval-viz renders with an index.
    Returns ``(combined_ref, combined_hyp)`` or None when there are no pairs.
    """
    if not pairs:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    combined_ref = dest_dir / "_combined.ref.stm"
    combined_hyp = dest_dir / "_combined.hyp.stm"
    combined_ref.write_text(
        "".join(_read_with_newline(ref) for _, ref, _ in pairs), encoding="utf-8"
    )
    combined_hyp.write_text(
        "".join(_read_with_newline(hyp) for _, _, hyp in pairs), encoding="utf-8"
    )
    return combined_ref, combined_hyp


def _read_with_newline(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def meeteval_viz_argv(
    ref_stm: Path,
    hyp_stm: Path,
    out_dir: Path,
    *,
    alignments: list[str],
) -> list[str]:
    """Build the ``meeteval-viz html`` argument vector (rendering stays meeteval's)."""
    return [
        sys.executable,
        "-m",
        "meeteval.viz",
        "html",
        "--alignment",
        *alignments,
        "-r",
        str(ref_stm),
        "-h",
        str(hyp_stm),
        "-o",
        str(out_dir),
    ]


def visualize_quality_dir(
    out_dir: Path,
    *,
    alignments: list[str] | None = None,
    viz_subdir: str = "viz",
) -> Path | None:
    """Render meeteval-viz HTML for every (ref, hyp) pair in a quality run.

    Returns the viz output directory, or None when no pairs are found.
    """
    alignments = alignments or ["tcp"]
    pairs = discover_quality_pairs(out_dir)
    combined = combine_session_stms(pairs, out_dir / viz_subdir)
    if combined is None:
        return None
    ref_stm, hyp_stm = combined
    viz_dir = out_dir / viz_subdir
    subprocess.run(  # noqa: S603
        meeteval_viz_argv(ref_stm, hyp_stm, viz_dir, alignments=alignments),
        check=True,
    )
    return viz_dir
