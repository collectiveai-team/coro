"""Published-WER calibration for the Spanish Workload Set.

Matching an externally published WER figure is the only available end-to-end
proof that the harness — decoding, **ASR Windowing**, **Hypothesis STM**
conversion, normalization and scoring — is free of systematic error. A large
deviation is therefore treated as a harness fault, not a model result, and fails
the run loudly.

Only citable, externally published figures are registered. When the configured
**ASR Model Selection** has no published figure for a calibration corpus, the
outcome is reported as ``unregistered`` and does not fail the run; there is
nothing to calibrate against and inventing a target would defeat the purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coro.bench.models.spanish import (
    CalibrationOutcome,
    CalibrationReport,
    CalibrationTarget,
    CorpusStats,
)
from coro.bench.spanish import SPANISH_CORPORA, corpus_of_item

DEFAULT_CALIBRATION_MARGIN = 0.10
"""Two-sided absolute WER band, in WER points expressed as a fraction.

Provisionally wide: the normalized lane currently only strips punctuation and
collapses whitespace, so casing differences still inflate it against published
figures. Tighten to ~0.05 once the diacritic-preserving basic normalizer lands
(perf-roadmap issue 06). The band is deliberately two-sided — a score far *below*
the published figure is as strong a harness-fault signal as one far above it.
"""

CALIBRATION_METRIC = "normalized_orcwer"
"""Diarization-invariant normalized WER lane used for calibration.

The Spanish corpora are single-speaker, so cpWER would additionally penalise any
speaker split the diarizer invents; ORC-WER isolates ASR quality.
"""

_WHISPER_PAPER = "Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision"
_WHISPER_PAPER_URL = "https://arxiv.org/abs/2212.04356"
_PARAKEET_CARD = "NVIDIA parakeet-tdt-0.6b-v3 model card, multilingual ASR table"
_PARAKEET_CARD_URL = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"

# Whisper figures are Tables 13 (Fleurs) and 10 (MLS) of the Whisper paper,
# Spanish column. Parakeet figures are the model card's per-language table.
PUBLISHED_WER: tuple[CalibrationTarget, ...] = (
    *(
        CalibrationTarget(
            model_id=f"openai/whisper-{size}",
            corpus=corpus,
            published_wer=value,
            source=f"{_WHISPER_PAPER} ({table})",
            source_url=_WHISPER_PAPER_URL,
        )
        for size, fleurs, mls in (
            ("tiny", 0.159, 0.192),
            ("base", 0.099, 0.128),
            ("small", 0.056, 0.078),
            ("medium", 0.036, 0.053),
            ("large", 0.035, 0.054),
            ("large-v2", 0.030, 0.042),
        )
        for corpus, value, table in (
            ("fleurs", fleurs, "Table 13"),
            ("mls", mls, "Table 10"),
        )
    ),
    CalibrationTarget(
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        corpus="fleurs",
        published_wer=0.0345,
        source=_PARAKEET_CARD,
        source_url=_PARAKEET_CARD_URL,
    ),
    CalibrationTarget(
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        corpus="mls",
        published_wer=0.0439,
        source=_PARAKEET_CARD,
        source_url=_PARAKEET_CARD_URL,
    ),
)


def find_target(model_id: str | None, corpus: str) -> CalibrationTarget | None:
    """Return the published figure for one model/corpus pair, if registered."""
    if not model_id:
        return None
    wanted = model_id.strip().lower()
    for target in PUBLISHED_WER:
        if target.model_id.lower() == wanted and target.corpus == corpus:
            return target
    return None


def _metric_stats(metrics: dict[str, Any] | None) -> tuple[int, int] | None:
    """Return ``(errors, length)`` for the calibration metric of one item."""
    normalized = (metrics or {}).get("normalized") or {}
    stats = normalized.get("orcwer")
    if not isinstance(stats, dict):
        return None
    errors = stats.get("errors")
    length = stats.get("length")
    if errors is None or length is None:
        return None
    return int(errors), int(length)


def collect_corpus_stats(quality_dir: Path) -> list[CorpusStats]:
    """Aggregate errors/length per Spanish corpus from ``quality/`` artifacts.

    Only items whose id prefix names a known Spanish corpus are considered, so
    AMI items in a mixed run are ignored rather than mis-attributed.
    """
    totals: dict[str, CorpusStats] = {}
    for path in sorted(quality_dir.glob("*.json")):
        if path.name in {"summary.json", "calibration.json"}:
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        session_id = str(artifact.get("session_id") or path.stem)
        corpus = corpus_of_item(session_id)
        if corpus not in SPANISH_CORPORA:
            continue
        stats = _metric_stats(artifact.get("metrics"))
        if stats is None:
            continue
        errors, length = stats
        bucket = totals.setdefault(corpus, CorpusStats(corpus=corpus))
        bucket.errors += errors
        bucket.length += length
        bucket.items += 1
    return [totals[key] for key in sorted(totals)]


def _outcome(
    bucket: CorpusStats,
    model_id: str | None,
    margin: float,
) -> CalibrationOutcome:
    corpus = bucket.corpus
    if bucket.length <= 0:
        return CalibrationOutcome(
            corpus=corpus,
            status="no-score",
            n_items=bucket.items,
            model_id=model_id,
            detail="No scored reference words for this corpus.",
        )

    scored = bucket.errors / bucket.length
    target = find_target(model_id, corpus)
    if target is None:
        return CalibrationOutcome(
            corpus=corpus,
            status="unregistered",
            n_items=bucket.items,
            scored_wer=round(scored, 6),
            model_id=model_id,
            detail=(
                f"No published {CALIBRATION_METRIC} figure is registered for "
                f"{model_id!r} on {corpus}; nothing to calibrate against."
            ),
        )

    deviation = scored - target.published_wer
    status = "pass" if abs(deviation) <= margin else "fail"
    return CalibrationOutcome(
        corpus=corpus,
        status=status,
        n_items=bucket.items,
        scored_wer=round(scored, 6),
        published_wer=target.published_wer,
        deviation=round(deviation, 6),
        margin=margin,
        model_id=model_id,
        source=target.source,
        source_url=target.source_url,
        detail=(
            ""
            if status == "pass"
            else (
                f"Scored {scored:.4f} against a published {target.published_wer:.4f} "
                f"(deviation {deviation:+.4f} > margin {margin:.4f}). Treat this as a "
                "harness fault until proven otherwise."
            )
        ),
    )


def calibrate_run(
    out_dir: Path,
    *,
    model_id: str | None,
    margin: float = DEFAULT_CALIBRATION_MARGIN,
) -> CalibrationReport:
    """Calibrate a completed benchmark run against published Spanish WER.

    Args:
        out_dir: The benchmark run output directory (contains ``quality/``).
        model_id: The configured ASR Model Selection reported by ``/health``.
        margin: Two-sided absolute WER band, in WER points as a fraction.

    Returns:
        A :class:`CalibrationReport`; ``failed`` is True when any registered
        corpus deviated beyond the margin.

    """
    quality_dir = out_dir / "quality"
    report = CalibrationReport(
        model_id=model_id,
        metric=CALIBRATION_METRIC,
        margin=margin,
    )
    if not quality_dir.is_dir():
        return report

    for bucket in collect_corpus_stats(quality_dir):
        report.outcomes.append(_outcome(bucket, model_id, margin))

    report.failed = any(outcome.status == "fail" for outcome in report.outcomes)
    return report


def model_id_from_manifest(out_dir: Path) -> str | None:
    """Return the ASR Model Selection recorded in a run's ``manifest.json``."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    health = manifest.get("server_health") or {}
    selection = health.get("startup_selection") or {}
    model = selection.get("asr_model")
    return str(model) if model else None


def render_calibration(report: CalibrationReport) -> str:
    """Render a calibration report as human-readable text."""
    if not report.outcomes:
        return ""
    lines = [
        "",
        f"Spanish WER calibration ({report.metric}, "
        f"margin ±{report.margin:.3f}, model {report.model_id or 'unknown'})",
    ]
    for outcome in report.outcomes:
        scored = "n/a" if outcome.scored_wer is None else f"{outcome.scored_wer:.4f}"
        published = "n/a" if outcome.published_wer is None else f"{outcome.published_wer:.4f}"
        lines.append(
            f"  [{outcome.status.upper():>12}] {outcome.corpus:<10} "
            f"scored={scored} published={published} items={outcome.n_items}"
        )
        if outcome.detail:
            lines.append(f"                 {outcome.detail}")
        if outcome.source:
            lines.append(f"                 source: {outcome.source} <{outcome.source_url}>")
    return "\n".join(lines)
