"""Quality Benchmark scoring: MeetEval Metric Set and run-level summary.

Every WER metric is reported in two lanes, per Workload Item and in the
run-level aggregate:

- **Raw Metric Lane** — Reference STM and Hypothesis STM text scored verbatim.
  Catches punctuation and casing regressions the normalizer hides.
- **Normalized Metric Lane** — both sides passed through the vendored Whisper
  Basic Text Normalizer first. This is the lane comparable to published WER.

Normalization is not neutral, so neither lane is reported alone. See ADR 0011.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from coro.bench.models.quality import (
    CombinedMetrics,
    DerStats,
    NormalizedMetrics,
    PerItemEntry,
    QualitySummary,
    ScoreError,
    ScoreMetrics,
    ScoreResult,
    WderStats,
    WerStats,
)
from coro.bench.quality_meeteval import (
    Meeteval,
    combine_der_sessions,
    combine_wer_sessions,
    der_stats,
    require_meeteval,
    wer_stats,
)
from coro.bench.quality_stm import (
    diarization_sanity,
    is_diarization_only_stm,
    normalize_transcript_text,
    write_normalized_stm,
)
from coro.bench.wder import combine_wder, compute_wder
from coro.core.logger import get_logger

if TYPE_CHECKING:
    from coro.bench.quality_meeteval import ErrorRate

__all__ = [
    "WER_METRIC_KEYS",
    "combine_items",
    "diarization_sanity",
    "is_diarization_only_stm",
    "normalize_transcript_text",
    "score_item",
]

log = get_logger(__name__)


def _score_normalized_lane(
    meeteval: Meeteval,
    ref_stm_path: Path,
    hyp_stm_path: Path,
    raw: dict[str, ErrorRate],
) -> NormalizedMetrics | None:
    """Score the Normalized Metric Lane, recording raw results under ``raw``.

    Returns ``None`` when normalization leaves either side with no scoreable
    lines. That is reported as a skipped lane rather than raised, so a
    degenerate normalized comparison cannot take the Raw Metric Lane down
    with it.

    Args:
        meeteval: The imported ``meeteval`` module.
        ref_stm_path: Reference STM path.
        hyp_stm_path: Hypothesis STM path.
        raw: Mutable map of raw meeteval results to extend.

    Returns:
        The Normalized Metric Lane metrics, or ``None`` if it was skipped.

    """
    with tempfile.TemporaryDirectory(prefix="coro-quality-") as tmp:
        tmp_dir = Path(tmp)
        normalized_ref = tmp_dir / ref_stm_path.name
        normalized_hyp = tmp_dir / hyp_stm_path.name
        n_ref = write_normalized_stm(ref_stm_path, normalized_ref)
        n_hyp = write_normalized_stm(hyp_stm_path, normalized_hyp)

        if not n_ref or not n_hyp:
            log.warning(
                "normalized lane skipped: too few lines survived normalization",
                item=hyp_stm_path.name,
                reference_lines=n_ref,
                hypothesis_lines=n_hyp,
            )
            return None

        # Keep the per-session dict: its CPErrorRate.assignment is the optimal
        # speaker mapping WDER needs, and combining discards it.
        cpwer_per_session = meeteval.wer.cpwer(normalized_ref, normalized_hyp)
        raw["normalized_cpwer"] = combine_wer_sessions(meeteval, cpwer_per_session)
        raw["normalized_orcwer"] = combine_wer_sessions(
            meeteval, meeteval.wer.greedy_orcwer(normalized_ref, normalized_hyp)
        )
        raw["normalized_dicpwer"] = combine_wer_sessions(
            meeteval, meeteval.wer.greedy_dicpwer(normalized_ref, normalized_hyp)
        )
        # Scored inside the temporary directory: compute_wder re-reads both STM
        # files from disk, so the normalized copies must still exist.
        wder = compute_wder(normalized_ref, normalized_hyp, cpwer_per_session)
        return NormalizedMetrics(
            cpwer=wer_stats(raw["normalized_cpwer"]),
            orcwer=wer_stats(raw["normalized_orcwer"]),
            dicpwer=wer_stats(raw["normalized_dicpwer"]),
            wder=wder,
        )


def score_item(
    ref_stm_path: Path,
    hyp_stm_path: Path,
    *,
    der_collar: float = 0.0,
    der_regions: str = "all",
) -> ScoreResult:
    """Score one hypothesis STM against the reference STM in both metric lanes.

    Passes file paths directly to meeteval so it handles STM parsing
    internally. The multifile API returns dict[session_id -> result];
    results are combined across sessions with combine_error_rates.

    Each WER metric is computed twice: once on the STM files as written (Raw
    Metric Lane) and once on normalized copies of both (Normalized Metric Lane).
    DER is lane-independent -- it scores speaker time, not text -- so it is
    computed once and belongs to neither lane.

    siWER (SISO-WER) is omitted because AMI data has multiple speakers
    per session, making (session, speaker) pairs non-unique — a hard
    requirement of siWER.

    DER scoring protocol: NIST ``md-eval-22.pl`` via meeteval, at a **0 s collar**
    with **overlapping speech scored** (``regions="all"``) and **no UEM**, so the
    whole recording counts. That matches the convention published diarization
    results for AMI use.

    What does *not* match: when the caller passes a hypothesis built by
    :func:`coro.bench.stm.hyp_segments_to_stm`, the hypothesis timeline is the
    ASR's segmentation, not the diarizer's, and those segments tile the audio with
    no silence and no overlap. Missed-speech and false-alarm are then properties of
    ASR segmentation and are near-invariant to the diarization model; only
    speaker-error responds to it. Use ``coro-bench-diar`` (which scores a real
    diarizer timeline) for model-vs-model diarization comparisons, and pass
    ``der_collar=0.0`` there to keep this protocol.
    """
    meeteval = require_meeteval()

    try:
        raw: dict[str, ErrorRate] = {}
        metrics = ScoreMetrics()

        # Diarization-only references (speaker turns, no transcript) can only be
        # scored for DER; computing WER against a sentinel transcript would be
        # meaningless, so it is skipped.
        diarization_only = is_diarization_only_stm(ref_stm_path)

        if not diarization_only:
            # Keep the per-session dict: its CPErrorRate.assignment is the
            # optimal speaker mapping WDER needs, and combining discards it.
            cpwer_per_session = meeteval.wer.cpwer(ref_stm_path, hyp_stm_path)
            raw["cpwer"] = combine_wer_sessions(meeteval, cpwer_per_session)
            metrics.cpwer = wer_stats(raw["cpwer"])

            metrics.wder = compute_wder(ref_stm_path, hyp_stm_path, cpwer_per_session)

            raw["orcwer"] = combine_wer_sessions(
                meeteval, meeteval.wer.greedy_orcwer(ref_stm_path, hyp_stm_path)
            )
            metrics.orcwer = wer_stats(raw["orcwer"])

            raw["dicpwer"] = combine_wer_sessions(
                meeteval, meeteval.wer.greedy_dicpwer(ref_stm_path, hyp_stm_path)
            )
            metrics.dicpwer = wer_stats(raw["dicpwer"])

            # Normalized Metric Lane: the same four metrics recomputed after
            # applying the Basic Text Normalizer to both sides identically.
            metrics.normalized = _score_normalized_lane(meeteval, ref_stm_path, hyp_stm_path, raw)

        der_results = meeteval.der.md_eval_22(
            ref_stm_path,
            hyp_stm_path,
            collar=der_collar,
            regions=der_regions,
        )
        raw_der = combine_der_sessions(meeteval, der_results)
        metrics.der = der_stats(raw_der)

        return ScoreResult(
            metrics=metrics,
            diarization_only=diarization_only,
            diarization=diarization_sanity(ref_stm_path, hyp_stm_path),
            raw=raw,
            raw_der=raw_der,
        )

    except Exception as exc:
        # Log the full traceback so the operator can see the real cause
        # without having to dig into the JSON artifact.
        log.exception("scoring failed", item=hyp_stm_path.name)
        return ScoreResult(
            metrics=None,
            error=ScoreError(type=type(exc).__name__, message=str(exc)),
        )


# sisower removed: SISO-WER requires unique (session, speaker) pairs,
# which AMI multi-speaker meetings do not satisfy.
WER_METRIC_KEYS = ("cpwer", "orcwer", "dicpwer")


def _combine_raw_key(
    meeteval: Meeteval, succeeded: list[ScoreResult], raw_key: str
) -> WerStats | None:
    """Aggregate one WER ``raw`` key across all succeeded items."""
    raw_objects = [r.raw[raw_key] for r in succeeded if raw_key in r.raw]
    if not raw_objects:
        return None
    return wer_stats(meeteval.wer.combine_error_rates(*raw_objects))


def _combine_raw_der(meeteval: Meeteval, succeeded: list[ScoreResult]) -> DerStats | None:
    """Aggregate DER across all succeeded items."""
    raw_objects = [r.raw_der for r in succeeded if r.raw_der is not None]
    if not raw_objects:
        return None
    return der_stats(meeteval.wer.combine_error_rates(*raw_objects))


def _collect_wder(succeeded: list[ScoreResult], *, normalized: bool) -> WderStats | None:
    """Pool the per-item WDER counts across items, skipping items without one."""
    stats: list[WderStats] = []
    for result in succeeded:
        metrics = result.metrics
        if metrics is None:
            continue
        block = metrics.normalized if normalized else metrics
        candidate = block.wder if block is not None else None
        if candidate is not None:
            stats.append(candidate)
    return combine_wder(stats)


def _combined_metrics(meeteval: Meeteval, succeeded: list[ScoreResult]) -> CombinedMetrics:
    """Build the workload-level combined metric block from succeeded items."""
    return CombinedMetrics(
        cpwer=_combine_raw_key(meeteval, succeeded, "cpwer"),
        orcwer=_combine_raw_key(meeteval, succeeded, "orcwer"),
        dicpwer=_combine_raw_key(meeteval, succeeded, "dicpwer"),
        wder=_collect_wder(succeeded, normalized=False),
        normalized=NormalizedMetrics(
            cpwer=_combine_raw_key(meeteval, succeeded, "normalized_cpwer"),
            orcwer=_combine_raw_key(meeteval, succeeded, "normalized_orcwer"),
            dicpwer=_combine_raw_key(meeteval, succeeded, "normalized_dicpwer"),
            wder=_collect_wder(succeeded, normalized=True),
        ),
        der=_combine_raw_der(meeteval, succeeded),
    )


def _rate(stats: WerStats | None) -> float | None:
    """Pull the headline rate out of an optional WER breakdown."""
    return stats.wer if stats is not None else None


def _fill_normalized(entry: PerItemEntry, normalized: NormalizedMetrics | None) -> None:
    """Copy the punctuation-normalized rates onto a summary row."""
    if normalized is None:
        return
    entry.normalized_cpwer = _rate(normalized.cpwer)
    entry.normalized_orcwer = _rate(normalized.orcwer)
    entry.normalized_dicpwer = _rate(normalized.dicpwer)
    if normalized.wder is not None:
        entry.normalized_wder = normalized.wder.wder


def _per_item_entry(result: ScoreResult) -> PerItemEntry:
    """Flatten one item's metrics + diarization sanity into a summary row."""
    entry = PerItemEntry(
        session_id=result.session_id,
        audio_seconds=result.audio_seconds,
        diarization_only=True if result.diarization_only else None,
        diarization=result.diarization,
    )
    metrics = result.metrics
    if metrics is None:
        return entry

    entry.cpwer = _rate(metrics.cpwer)
    entry.orcwer = _rate(metrics.orcwer)
    entry.dicpwer = _rate(metrics.dicpwer)
    if metrics.der is not None:
        entry.der = metrics.der.der
    if metrics.wder is not None:
        entry.wder = metrics.wder.wder
        entry.wder_claimed = metrics.wder.wder_claimed
        entry.abstention_rate = metrics.wder.abstention_rate
    _fill_normalized(entry, metrics.normalized)
    return entry


def combine_items(item_results: list[ScoreResult]) -> QualitySummary:
    """Aggregate per-item score_item results into a workload-level summary."""
    meeteval = require_meeteval()

    succeeded = [r for r in item_results if r.metrics is not None]
    failed = [r for r in item_results if r.metrics is None]
    n_degenerate = sum(
        1 for r in item_results if r.diarization is not None and r.diarization.degenerate
    )

    return QualitySummary(
        workload_set=[r.session_id for r in item_results],
        n_succeeded=len(succeeded),
        n_failed=len(failed),
        n_degenerate_diarization=n_degenerate,
        combined=_combined_metrics(meeteval, succeeded),
        per_item=[_per_item_entry(r) for r in item_results],
    )
