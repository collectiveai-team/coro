"""Spanish Workload Set preparation and WER calibration for the coro-bench CLI.

Kept separate from the run orchestration in :mod:`coro.bench.cli` for the same
reason the flag surface lives in :mod:`coro.bench.cli_args`: materialising a
public corpus and scoring the run against published WER are their own concern,
not part of driving a benchmark.
"""

from __future__ import annotations

import argparse
import sys


def print_spanish_fetch_plan(args: argparse.Namespace) -> None:
    """Print the per-corpus download footprint of the selected Spanish preset."""
    from coro.bench.spanish import preset_fetch_plan

    plans = preset_fetch_plan(args.spanish_preset, items_per_corpus=args.spanish_limit)
    print(f"Spanish preset '{args.spanish_preset}' fetch plan:")
    total = 0
    for plan in plans:
        total += plan.download_bytes
        print(
            f"  {plan.corpus:<12} rows={plan.rows:<5} "
            f"row_groups={plan.row_groups:<3} download≈{plan.download_bytes / 1e6:,.1f} MB"
        )
    print(f"  {'total':<12} download≈{total / 1e6:,.1f} MB (one-time; cached on disk)")


def prepare_spanish_workload(args: argparse.Namespace) -> None:
    """Materialise the selected Spanish preset into ``args.clips_dir``."""
    from coro.bench.spanish import materialize_spanish_workload_set

    args.clips_dir = materialize_spanish_workload_set(
        args.spanish_preset,
        args.spanish_root,
        items_per_corpus=args.spanish_limit,
        no_download=args.no_download,
    )
    print(f"Spanish workload set materialised at {args.clips_dir}")


def run_calibration(args: argparse.Namespace) -> bool:
    """Score the run against published WER; return True when it deviates."""
    import json
    from dataclasses import asdict

    from coro.bench.calibration import calibrate_run, model_id_from_manifest, render_calibration

    report = calibrate_run(
        args.out_dir,
        model_id=model_id_from_manifest(args.out_dir),
        margin=args.calibration_margin,
    )
    if not report.outcomes:
        return False

    print(render_calibration(report))
    quality_dir = args.out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "calibration.json").write_text(json.dumps(asdict(report), indent=2))

    if report.failed and not args.no_calibration:
        print(
            "error: Spanish WER calibration deviated beyond the margin; "
            "treat this as a harness fault until proven otherwise.",
            file=sys.stderr,
        )
        return True
    return False
