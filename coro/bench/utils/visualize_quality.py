#!/usr/bin/env python3

r"""Render side-by-side reference/hypothesis alignment HTML for a quality run.

Convenience over meeteval's own ``meeteval-viz html``: discovers the per-session
(ref, hyp) STM pairs written by ``coro-bench quality`` / ``all`` under
``<out-dir>/ref/<id>.ref.stm`` and ``<out-dir>/hyp/<id>.hyp.stm``, and renders
them in one call under ``<out-dir>/viz/`` (index + per-session + the synced
``side_by_side_sync.html`` when multiple alignments are given).

Run from the environment where this package and meeteval are installed (e.g. the
project venv: ``.venv/bin/python -m ...``); ``uv run`` from a different repo may
resolve a venv that lacks meeteval's viz extras (simplejson).

    coro-bench quality --clips-dir clips --server-url ... --out-dir OUT
    python -m coro.bench.utils.visualize_quality OUT --alignment tcp cp

To visualise a single pair directly, call meeteval-viz on the discovered files
(note hypotheses are named ``<id>.hyp.stm``):

    meeteval-viz html --alignment tcp cp \\
        -r OUT/ref/IB4001_180_60.ref.stm \\
        -h OUT/hyp/IB4001_180_60.hyp.stm -o OUT/viz
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from coro.bench.viz import visualize_quality_dir
from coro.cli_support import invoke, new_app, out_console

app = new_app("visualize_quality", help="Render ref/hyp alignment HTML from a quality run out-dir.")


@app.default
def _render(
    out_dir: Path,
    /,
    *,
    alignment: Annotated[
        list[str] | None,
        Parameter(
            consume_multiple=True,
            help="meeteval alignment algorithm(s); multiple enable the synced "
            "side-by-side comparison (e.g. tcp cp). Default: tcp.",
        ),
    ] = None,
) -> None:
    """Render ref/hyp alignment HTML from a quality run out-dir.

    Args:
        out_dir: Quality run output directory.
        alignment: meeteval alignment algorithm(s).

    """
    viz_dir = visualize_quality_dir(out_dir, alignments=alignment or ["tcp"])
    if viz_dir is None:
        out_console.out(f"No (ref, hyp) STM pairs found under {out_dir}.")
        return
    out_console.out(f"wrote visualizations to {viz_dir} (open {viz_dir / 'index.html'})")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the alignment visualiser command line."""
    invoke(app, argv)


if __name__ == "__main__":
    main()
