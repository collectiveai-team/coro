"""Quality scoring step of a benchmark run: per-item and summary JSON artifacts.

Scores each Workload Item's ``hyp/`` STM against its ``ref/`` copy and writes
``quality/<item>.json`` plus ``quality/summary.json``. ``score_item`` and
``combine_items`` are looked up on :mod:`coro.bench.quality` at call time so
tests can patch them there.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from coro.bench.models.quality import ScoreResult
    from coro.bench.models.workload import WorkloadItem


class _ItemArtifact(TypedDict, total=False):
    session_id: str
    audio_seconds: float
    metrics: Mapping[str, object] | None
    diarization: Mapping[str, object]
    error: Mapping[str, object]


def _score_and_write(
    item: WorkloadItem,
    out_dir: Path,
    *,
    der_collar: float,
    der_regions: str,
    with_diarization: bool,
) -> ScoreResult:
    """Score one item, write its ``quality/<item>.json`` artifact and return the score."""
    from coro.bench.quality import score_item

    item_id = item["item_id"]
    scored = score_item(
        out_dir / "ref" / f"{item_id}.ref.stm",
        out_dir / "hyp" / f"{item_id}.hyp.stm",
        der_collar=der_collar,
        der_regions=der_regions,
    )
    scored.session_id = item_id
    scored.audio_seconds = item.get("audio_seconds", 0.0)

    artifact: _ItemArtifact = {
        "session_id": item_id,
        "audio_seconds": item.get("audio_seconds", 0.0),
        "metrics": asdict(scored.metrics) if scored.metrics is not None else None,
    }
    if with_diarization and scored.diarization is not None:
        artifact["diarization"] = asdict(scored.diarization)
    if scored.error is not None:
        artifact["error"] = asdict(scored.error)

    (out_dir / "quality" / f"{item_id}.json").write_text(json.dumps(artifact, indent=2))
    return scored


def run_quality_scoring_with_skip(
    *,
    out_dir: Path,
    items: list[WorkloadItem],
    der_collar: float,
    der_regions: str,
) -> None:
    """Score the items that have a Reference STM and count the ones that do not."""
    from coro.bench.models.quality import QualitySummary
    from coro.bench.quality import combine_items

    quality_dir = out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    item_results: list[ScoreResult] = []
    n_skipped = 0
    for item in items:
        if item.get("ref_stm_path") is None:
            n_skipped += 1
            continue
        item_results.append(
            _score_and_write(
                item,
                out_dir,
                der_collar=der_collar,
                der_regions=der_regions,
                with_diarization=True,
            )
        )

    summary = combine_items(item_results) if item_results else QualitySummary()
    summary.n_skipped = n_skipped
    (quality_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))


def run_quality_scoring(
    *,
    out_dir: Path,
    items: list[WorkloadItem],
    der_collar: float,
    der_regions: str,
) -> None:
    """Score every item; one without a Reference STM is recorded as a failed score."""
    from coro.bench.quality import combine_items

    quality_dir = out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    item_results = [
        _score_and_write(
            item,
            out_dir,
            der_collar=der_collar,
            der_regions=der_regions,
            with_diarization=False,
        )
        for item in items
    ]

    summary = combine_items(item_results)
    (quality_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
