"""Spanish Workload Set preparation and WER calibration for the coro-bench CLI.

Kept separate from the run orchestration in :mod:`coro.bench.cli` for the same
reason the flag surface lives in :mod:`coro.bench.cli_args`: materialising a
public corpus and scoring the run against published WER are their own concern,
not part of driving a benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coro.cli_support import err_console, out_console
from coro.core.logger import get_logger

if TYPE_CHECKING:
    from coro.bench.cli_flags import BenchArgs

log = get_logger(__name__)


def print_spanish_fetch_plan(args: BenchArgs) -> None:
    """Print the per-corpus download footprint of the selected Spanish preset."""
    from coro.bench.spanish import preset_fetch_plan

    preset = args.spanish_preset
    if preset is None:
        return
    plans = preset_fetch_plan(preset, items_per_corpus=args.spanish_limit)
    out_console.out(f"Spanish preset '{preset}' fetch plan:")
    total = 0
    for plan in plans:
        total += plan.download_bytes
        out_console.out(
            f"  {plan.corpus:<12} rows={plan.rows:<5} "
            f"row_groups={plan.row_groups:<3} download≈{plan.download_bytes / 1e6:,.1f} MB"
        )
    out_console.out(f"  {'total':<12} download≈{total / 1e6:,.1f} MB (one-time; cached on disk)")


def prepare_spanish_workload(args: BenchArgs) -> None:
    """Materialise the selected Spanish preset into ``args.clips_dir``."""
    from coro.bench.spanish import materialize_spanish_workload_set

    preset = args.spanish_preset
    if preset is None:
        return
    args.clips_dir = materialize_spanish_workload_set(
        preset,
        args.spanish_root,
        items_per_corpus=args.spanish_limit,
        no_download=args.no_download,
    )
    log.info("spanish workload set materialised", clips_dir=str(args.clips_dir))


def run_calibration(args: BenchArgs) -> bool:
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

    out_console.out(render_calibration(report))
    quality_dir = args.out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "calibration.json").write_text(json.dumps(asdict(report), indent=2))

    if report.failed and not args.no_calibration:
        err_console.out(
            "error: Spanish WER calibration deviated beyond the margin; "
            "treat this as a harness fault until proven otherwise."
        )
        return True
    return False
