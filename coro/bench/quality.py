"""Quality Benchmark scoring: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import sys
import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from coro.bench.text import TEXT_SCHEMAS, write_schema_stm
from coro.bench.models.quality import (
    CombinedMetrics,
    DerStats,
    DiarizationSanity,
    SchemaMetrics,
    PerItemEntry,
    QualitySummary,
    ScoreError,
    ScoreMetrics,
    ScoreResult,
    SegmentShapeCounters,
    WerStats,
)


def _require_meeteval():
    try:
        import meeteval

        return meeteval
    except ImportError:
        print(
            "Error: meeteval is required for quality scoring.\n"
            "Install with: pip install coro[bench]",
            file=sys.stderr,
        )
        sys.exit(1)


def _wer_to_dict(result) -> WerStats:
    """Convert a meeteval WER result object to :class:`WerStats`.

    meeteval 0.4.x uses `error_rate` instead of `wer` as the attribute name.
    """
    return WerStats(
        wer=result.error_rate,
        errors=result.errors,
        length=result.length,
        insertions=result.insertions,
        deletions=result.deletions,
        substitutions=result.substitutions,
    )


def _der_to_dict(result) -> DerStats:
    """Convert a meeteval DER result object to :class:`DerStats`.

    meeteval 0.4.x DiaErrorRate uses `*_speaker_time` field names and
    returns Decimal values; cast to float for JSON serialisability.
    """
    return DerStats(
        der=float(result.error_rate),
        false_alarm=float(result.falarm_speaker_time),
        missed_detection=float(result.missed_speaker_time),
        speaker_error=float(result.speaker_error_time),
        total_speech=float(result.scored_speaker_time),
    )


def _combine_multifile(meeteval, results: dict) -> Any:
    """Combine per-session meeteval results into a single aggregate."""
    return meeteval.wer.combine_error_rates(*results.values())


def is_diarization_only_stm(path: Path) -> bool:
    """Return True when every STM line's text is the diarization-only sentinel.

    Such references (e.g. VoxConverse RTTM converted via rttm_to_stm) carry
    speaker turns but no transcript, so only DER is meaningful.
    """
    from coro.bench.stm import DIARIZATION_ONLY_TEXT

    saw_line = False
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        saw_line = True
        if parts[5].strip() != DIARIZATION_ONLY_TEXT:
            return False
    return saw_line


def stm_words_per_segment(path: Path) -> list[int]:
    """Return the word count of every segment in an STM file, in file order.

    The raw material for :func:`segment_shape_counters`. Kept separate so a
    workload-level aggregate can pool every item's segments before reducing,
    matching how meeteval combines raw error counts across items.
    """
    counts: list[int] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        counts.append(len(parts[5].split()))
    return counts


def segment_shape_counters(words_per_segment: Sequence[int]) -> SegmentShapeCounters:
    """Reduce per-segment word counts to Segment Shape Counters."""
    if not words_per_segment:
        return SegmentShapeCounters()
    return SegmentShapeCounters(
        segment_count=len(words_per_segment),
        median_words_per_segment=float(median(words_per_segment)),
        single_word_segment_count=sum(1 for n in words_per_segment if n == 1),
    )


def _count_stm_speakers(path: Path) -> int:
    """Count distinct speaker labels (column 3) in an STM file."""
    speakers: set[str] = set()
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=5)
        if len(parts) >= 3:
            speakers.add(parts[2])
    return len(speakers)


def diarization_sanity(ref_stm_path: Path, hyp_stm_path: Path) -> DiarizationSanity:
    """Flag degenerate diarization: a single hyp speaker against multi-speaker ref.

    A diarization-invariant metric like ORC-WER stays low even when every word
    collapses onto one speaker, so this check surfaces the failure the WER
    headline would otherwise hide.
    """
    ref_speakers = _count_stm_speakers(ref_stm_path)
    hyp_speakers = _count_stm_speakers(hyp_stm_path)
    degenerate = hyp_speakers <= 1 and ref_speakers > 1
    return DiarizationSanity(
        ref_speakers=ref_speakers,
        hyp_speakers=hyp_speakers,
        degenerate=degenerate,
    )


def _score_wer_triple(meeteval, ref: Path, hyp: Path, raw: dict[str, Any], prefix: str) -> None:
    """Score the three WER metrics for one (ref, hyp) pair into ``raw``.

    ``prefix`` namespaces the raw keys so a text schema's results sit beside the
    unnormalized ones without colliding. An empty prefix means raw text.
    """
    scorers = {
        "cpwer": meeteval.wer.cpwer,
        "orcwer": meeteval.wer.greedy_orcwer,
        "dicpwer": meeteval.wer.greedy_dicpwer,
    }
    for metric, scorer in scorers.items():
        key = f"{prefix}_{metric}" if prefix else metric
        raw[key] = _combine_multifile(meeteval, scorer(ref, hyp))


def _metrics_from_raw(raw: dict[str, Any], prefix: str) -> SchemaMetrics:
    """Collect one text schema's raw results into a metric block."""
    return SchemaMetrics(
        cpwer=_wer_to_dict(raw[f"{prefix}_cpwer"]),
        orcwer=_wer_to_dict(raw[f"{prefix}_orcwer"]),
        dicpwer=_wer_to_dict(raw[f"{prefix}_dicpwer"]),
    )


def _score_text_schemas(meeteval, ref: Path, hyp: Path, raw: dict[str, Any]) -> None:
    """Score every text schema into ``raw``, each from its own rewritten STMs.

    One loop over the schema registry, so a new schema is an entry there plus a
    field on the metric model — never another copy of the scoring block.
    """
    with tempfile.TemporaryDirectory(prefix="coro-quality-") as tmp:
        for prefix, normalize in TEXT_SCHEMAS:
            schema_dir = Path(tmp) / prefix
            schema_dir.mkdir()
            schema_ref = schema_dir / ref.name
            schema_hyp = schema_dir / hyp.name
            write_schema_stm(ref, schema_ref, normalize)
            write_schema_stm(hyp, schema_hyp, normalize)
            _score_wer_triple(meeteval, schema_ref, schema_hyp, raw, prefix)


def score_item(
    ref_stm_path: Path,
    hyp_stm_path: Path,
    *,
    der_collar: float = 0.0,
    der_regions: str = "all",
) -> ScoreResult:
    """Score one hypothesis STM against the reference STM.

    Passes file paths directly to meeteval so it handles STM parsing
    internally. The multifile API returns dict[session_id -> result];
    results are combined across sessions with combine_error_rates.

    siWER (SISO-WER) is omitted because AMI data has multiple speakers
    per session, making (session, speaker) pairs non-unique — a hard
    requirement of siWER.
    """
    meeteval = _require_meeteval()

    try:
        raw: dict[str, Any] = {}
        metrics = ScoreMetrics()

        # Diarization-only references (speaker turns, no transcript) can only be
        # scored for DER; computing WER against a sentinel transcript would be
        # meaningless, so it is skipped.
        diarization_only = is_diarization_only_stm(ref_stm_path)

        if not diarization_only:
            _score_wer_triple(meeteval, ref_stm_path, hyp_stm_path, raw, "")
            metrics.cpwer = _wer_to_dict(raw["cpwer"])
            metrics.orcwer = _wer_to_dict(raw["orcwer"])
            metrics.dicpwer = _wer_to_dict(raw["dicpwer"])

            _score_text_schemas(meeteval, ref_stm_path, hyp_stm_path, raw)
            metrics.unpunctuated = _metrics_from_raw(raw, "unpunctuated")
            metrics.whisper_english = _metrics_from_raw(raw, "whisper_english")

        der_results = meeteval.der.md_eval_22(
            ref_stm_path,
            hyp_stm_path,
            collar=der_collar,
            regions=der_regions,
        )
        raw["der"] = _combine_multifile(meeteval, der_results)
        metrics.der = _der_to_dict(raw["der"])

        return ScoreResult(
            metrics=metrics,
            diarization_only=diarization_only,
            diarization=diarization_sanity(ref_stm_path, hyp_stm_path),
            raw=raw,
            segment_word_counts=stm_words_per_segment(hyp_stm_path),
        )

    except Exception as exc:
        # Print full traceback to stderr so the operator can see the real cause
        # without having to dig into the JSON artifact.
        print(
            f"[bench/quality] scoring failed for {hyp_stm_path.name}:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return ScoreResult(
            metrics=None,
            error=ScoreError(type=type(exc).__name__, message=str(exc)),
        )


# sisower removed: SISO-WER requires unique (session, speaker) pairs,
# which AMI multi-speaker meetings do not satisfy.
WER_METRIC_KEYS = ("cpwer", "orcwer", "dicpwer")


def _combine_raw_key(meeteval, succeeded: list[ScoreResult], raw_key: str) -> Any:
    """Aggregate one ``raw`` key (WER or DER) across all succeeded items."""
    raw_objects = [r.raw[raw_key] for r in succeeded if raw_key in r.raw]
    if not raw_objects:
        return None
    combined = meeteval.wer.combine_error_rates(*raw_objects)
    converter = _der_to_dict if raw_key == "der" else _wer_to_dict
    return converter(combined)


def _combined_schema(meeteval, succeeded: list[ScoreResult], prefix: str) -> SchemaMetrics:
    """Combine one text schema's WER triple across all succeeded items."""
    return SchemaMetrics(
        cpwer=_combine_raw_key(meeteval, succeeded, f"{prefix}_cpwer"),
        orcwer=_combine_raw_key(meeteval, succeeded, f"{prefix}_orcwer"),
        dicpwer=_combine_raw_key(meeteval, succeeded, f"{prefix}_dicpwer"),
    )


def _combined_metrics(meeteval, succeeded: list[ScoreResult]) -> CombinedMetrics:
    """Build the workload-level combined metric block from succeeded items."""
    return CombinedMetrics(
        cpwer=_combine_raw_key(meeteval, succeeded, "cpwer"),
        orcwer=_combine_raw_key(meeteval, succeeded, "orcwer"),
        dicpwer=_combine_raw_key(meeteval, succeeded, "dicpwer"),
        unpunctuated=_combined_schema(meeteval, succeeded, "unpunctuated"),
        whisper_english=_combined_schema(meeteval, succeeded, "whisper_english"),
        der=_combine_raw_key(meeteval, succeeded, "der"),
    )


def _per_item_entry(result: ScoreResult) -> PerItemEntry:
    """Flatten one item's metrics + diarization sanity into a summary row."""
    entry = PerItemEntry(
        session_id=result.session_id,
        audio_seconds=result.audio_seconds,
        diarization_only=True if result.diarization_only else None,
        diarization=result.diarization,
        segment_shape=segment_shape_counters(result.segment_word_counts),
    )
    metrics = result.metrics
    if metrics is not None:
        if metrics.cpwer is not None:
            entry.cpwer = metrics.cpwer.wer
        if metrics.orcwer is not None:
            entry.orcwer = metrics.orcwer.wer
        if metrics.dicpwer is not None:
            entry.dicpwer = metrics.dicpwer.wer
        if metrics.der is not None:
            entry.der = metrics.der.der
        for prefix, block in (
            ("unpunctuated", metrics.unpunctuated),
            ("whisper_english", metrics.whisper_english),
        ):
            if block is None:
                continue
            for metric in ("cpwer", "orcwer", "dicpwer"):
                stats = getattr(block, metric)
                if stats is not None:
                    setattr(entry, f"{prefix}_{metric}", stats.wer)
    return entry


def combine_items(item_results: list[ScoreResult]) -> QualitySummary:
    """Aggregate per-item score_item results into a workload-level summary."""
    meeteval = _require_meeteval()

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
        segment_shape=segment_shape_counters(
            [n for r in item_results for n in r.segment_word_counts]
        ),
    )
