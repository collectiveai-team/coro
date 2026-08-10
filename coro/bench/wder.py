"""WDER — Word Diarization Error Rate.

Shafey, Soltau & Shafran (2019), *Joint Speech Recognition and Speaker
Diarization via Sequence Transduction*; the metric DiarizationLM
(arXiv:2401.03506) uses to score the ASR-to-diarization reconciliation step.

``WDER = (S_IS + C_IS) / (S + C)`` where ``S`` is the number of ASR
substitutions, ``C`` the number of correct ASR words, and the ``_IS`` suffix
counts those carrying an *incorrect speaker* after the optimal hypothesis to
reference speaker mapping.

Why this metric exists here: insertions and deletions are excluded from the
denominator, so WDER is not diluted by an ASR error floor, and it is blind to
segmentation — re-chunking segments without moving any word label moves WDER by
exactly zero. cpWER can express neither property.

Two pieces are needed and both are taken from ``meeteval`` rather than
reimplemented:

1. the speaker mapping, from :class:`meeteval.wer.wer.cp.CPErrorRate.assignment`
   (see :func:`hyp_to_ref_speaker_map`);
2. the word alignment, from ``meeteval.wer.wer.time_constrained.align`` driven
   with a collar wider than the recording, which degenerates its
   time-constrained Levenshtein to a plain one. This is the same reuse
   ``meeteval.viz`` performs for its ``'levenshtein'`` alignment type, so no
   extra dependency and no hand-written backtrace is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coro.bench.models.quality import WderStats

UNKNOWN_SPEAKER_LABEL = "-1"
"""Hypothesis speaker sentinel meaning "no diarization support for this word".

Emitted by ``coro.core.response`` and carried through the Hypothesis STM. It is
never mapped onto a reference speaker: a word carrying it is an abstention, and
counts as an error in ``wder`` while being excluded from ``wder_claimed``.
"""


@dataclass(frozen=True)
class SpeakerMap:
    """Optimal hypothesis -> reference speaker mapping for one session."""

    pairs: dict[str, str] = field(default_factory=dict)

    def matches(self, hyp_speaker: str, ref_speaker: str) -> bool:
        """Report whether this hypothesis speaker maps onto that reference speaker.

        A hypothesis speaker absent from the mapping is unmatched — a surplus
        stream or the unknown sentinel — so it can never match.
        """
        return self.pairs.get(hyp_speaker) == ref_speaker


@dataclass(frozen=True)
class _Counts:
    """Additive tally behind the three reported rates."""

    correct: int = 0
    substitutions: int = 0
    speaker_errors: int = 0
    claimed: int = 0
    claimed_speaker_errors: int = 0
    abstentions: int = 0

    def __add__(self, other: _Counts) -> _Counts:
        return _Counts(
            correct=self.correct + other.correct,
            substitutions=self.substitutions + other.substitutions,
            speaker_errors=self.speaker_errors + other.speaker_errors,
            claimed=self.claimed + other.claimed,
            claimed_speaker_errors=self.claimed_speaker_errors + other.claimed_speaker_errors,
            abstentions=self.abstentions + other.abstentions,
        )


def hyp_to_ref_speaker_map(cp_result: Any) -> SpeakerMap:
    """Invert a meeteval cpWER ``assignment`` into hypothesis -> reference.

    ``CPErrorRate.assignment`` is a tuple of ``(ref_speaker, hyp_speaker)``
    pairs under the cost-minimising permutation, padded with ``None`` on either
    side when the stream counts differ. Unmatched streams and the unknown
    sentinel are dropped, so a lookup miss means "this hypothesis speaker maps
    to no reference speaker" and is therefore always a speaker error.
    """
    pairs: dict[str, str] = {}
    for ref_speaker, hyp_speaker in cp_result.assignment or ():
        if ref_speaker is None or hyp_speaker is None:
            continue
        if str(hyp_speaker) == UNKNOWN_SPEAKER_LABEL:
            continue
        pairs[str(hyp_speaker)] = str(ref_speaker)
    return SpeakerMap(pairs)


def _flat_word_alignment(reference: Any, hypothesis: Any) -> list[tuple]:
    """Align one session's reference and hypothesis words as two flat streams.

    WDER is defined over a single time-ordered word sequence per side, not over
    per-speaker streams: a word given to the wrong speaker must remain a
    substitution or a correct word so it can be counted as a speaker error. The
    per-stream alignment cpWER performs would turn it into an insertion plus a
    deletion, which WDER excludes, and the metric would read zero.
    """
    from meeteval.wer.wer.time_constrained import align

    segments = list(reference) + list(hypothesis)
    if not segments:
        return []

    # A collar wider than the recording makes every word overlap every other
    # word, which disables the time constraint and leaves a plain Levenshtein.
    # Timestamps stay in their native type (meeteval parses STM times as
    # Decimal) because meeteval subtracts the collar from them directly.
    span = max(s["end_time"] for s in segments) - min(s["start_time"] for s in segments)

    return align(
        reference,
        hypothesis,
        collar=span * 2 + 1,
        reference_pseudo_word_level_timing="equidistant_intervals",
        hypothesis_pseudo_word_level_timing="equidistant_intervals",
        reference_sort="segment",
        hypothesis_sort="segment",
        style="seglst",
    )


def _tally_word(ref_word: dict, hyp_word: dict, speakers: SpeakerMap) -> _Counts:
    """Classify one aligned reference/hypothesis word pair."""
    text_match = ref_word["words"] == hyp_word["words"]
    correct = 1 if text_match else 0
    substitutions = 0 if text_match else 1

    hyp_speaker = str(hyp_word["speaker"])
    if hyp_speaker == UNKNOWN_SPEAKER_LABEL:
        return _Counts(
            correct=correct,
            substitutions=substitutions,
            speaker_errors=1,
            abstentions=1,
        )

    wrong = 0 if speakers.matches(hyp_speaker, str(ref_word["speaker"])) else 1
    return _Counts(
        correct=correct,
        substitutions=substitutions,
        speaker_errors=wrong,
        claimed=1,
        claimed_speaker_errors=wrong,
    )


def _tally(alignment: list[tuple], speakers: SpeakerMap) -> _Counts:
    """Count correct/substituted words and their speaker errors."""
    total = _Counts()
    for ref_word, hyp_word in alignment:
        # Insertions and deletions carry no reference/hypothesis pair, so they
        # are outside both the numerator and the denominator by definition.
        if ref_word is None or hyp_word is None:
            continue
        total = total + _tally_word(ref_word, hyp_word, speakers)
    return total


def _stats_from_counts(counts: _Counts) -> WderStats:
    """Build a :class:`WderStats` from raw counts, guarding empty denominators.

    ``wder_claimed`` is ``None`` — undefined — rather than ``0.0`` when nothing
    was claimed; a system that abstains everywhere has no precision to report,
    and reporting zero would read as perfect attribution.
    """
    scored = counts.correct + counts.substitutions
    claimed = counts.claimed
    return WderStats(
        wder=(counts.speaker_errors / scored) if scored else None,
        wder_claimed=(counts.claimed_speaker_errors / claimed) if claimed else None,
        abstention_rate=(counts.abstentions / scored) if scored else None,
        scored=scored,
        speaker_errors=counts.speaker_errors,
        claimed=claimed,
        claimed_speaker_errors=counts.claimed_speaker_errors,
        abstentions=counts.abstentions,
        correct=counts.correct,
        substitutions=counts.substitutions,
    )


def compute_wder(
    ref_stm_path: Path,
    hyp_stm_path: Path,
    cp_results: dict[str, Any],
) -> WderStats:
    """Score WDER for one reference/hypothesis STM pair.

    ``cp_results`` is meeteval's per-session ``cpwer`` result dict, taken from
    the call the benchmark already makes; its ``assignment`` supplies the
    speaker mapping so the permutation is never recomputed here.
    """
    import meeteval

    reference = meeteval.io.load(ref_stm_path).to_seglst()
    hypothesis = meeteval.io.load(hyp_stm_path).to_seglst()

    ref_sessions = reference.groupby("session_id")
    hyp_sessions = hypothesis.groupby("session_id")

    totals = _Counts()
    for session_id, cp_result in cp_results.items():
        ref_session = ref_sessions.get(session_id)
        hyp_session = hyp_sessions.get(session_id)
        if ref_session is None or hyp_session is None:
            continue
        alignment = _flat_word_alignment(ref_session, hyp_session)
        totals = totals + _tally(alignment, hyp_to_ref_speaker_map(cp_result))

    return _stats_from_counts(totals)


def combine_wder(results: list[WderStats]) -> WderStats | None:
    """Pool per-item WDER results by summing counts, not by averaging rates.

    Matches how meeteval combines error rates across sessions: the workload
    value is the ratio of pooled numerator to pooled denominator, so long items
    weigh proportionally to their scored word count.
    """
    if not results:
        return None
    totals = _Counts(
        correct=sum(r.correct for r in results),
        substitutions=sum(r.substitutions for r in results),
        speaker_errors=sum(r.speaker_errors for r in results),
        claimed=sum(r.claimed for r in results),
        claimed_speaker_errors=sum(r.claimed_speaker_errors for r in results),
        abstentions=sum(r.abstentions for r in results),
    )
    return _stats_from_counts(totals)
