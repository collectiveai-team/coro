#!/usr/bin/env python3

r"""Render side-by-side reference/hypothesis alignment HTML for a quality run.

Convenience over meeteval's own ``meeteval-viz html``: discovers the per-session
(ref, hyp) STM pairs written by ``asr-diar-bench quality`` / ``all`` under
``<out-dir>/ref/<id>.ref.stm`` and ``<out-dir>/hyp/<id>.hyp.stm``, and renders
them in one call under ``<out-dir>/viz/`` (index + per-session + the synced
``side_by_side_sync.html`` when multiple alignments are given).

Run from the environment where this package and meeteval are installed (e.g. the
project venv: ``.venv/bin/python -m ...``); ``uv run`` from a different repo may
resolve a venv that lacks meeteval's viz extras (simplejson).

    asr-diar-bench quality --clips-dir clips --server-url ... --out-dir OUT
    python -m asr_diar_server.bench.utils.visualize_quality OUT --alignment tcp cp

To visualise a single pair directly, call meeteval-viz on the discovered files
(note hypotheses are named ``<id>.hyp.stm``):

    meeteval-viz html --alignment tcp cp \\
        -r OUT/ref/RNE14-es_0_120.ref.stm \\
        -h OUT/hyp/RNE14-es_0_120.hyp.stm -o OUT/viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_diar_server.bench.viz import visualize_quality_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ref/hyp alignment HTML from a quality run out-dir.",
    )
    parser.add_argument("out_dir", type=Path, help="Quality run output directory.")
    parser.add_argument(
        "--alignment",
        nargs="+",
        default=["tcp"],
        help="meeteval alignment algorithm(s); multiple enable the synced "
        "side-by-side comparison (e.g. tcp cp). Default: tcp.",
    )
    args = parser.parse_args()

    viz_dir = visualize_quality_dir(args.out_dir, alignments=args.alignment)
    if viz_dir is None:
        print(f"No (ref, hyp) STM pairs found under {args.out_dir}.")
        return
    print(f"wrote visualizations to {viz_dir} (open {viz_dir / 'index.html'})")


if __name__ == "__main__":
    main()
