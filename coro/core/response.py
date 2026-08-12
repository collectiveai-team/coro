"""Core response builder.

Accepts Project-Owned Transcript Model types and produces the enriched
:class:`TranscriptionResult`.  No FastAPI or backend-native types are used.

Key behaviours:
- Groups tokens into segment runs using the Spanish-aware policy in
  :mod:`coro.core.segmentation`. Segment boundaries are *sentence-first*: they
  come from the transcript alone and are never cut at a word-level speaker
  change.
- Assigns a speaker to every *word* from the diarization timeline. Those labels
  are the normative per-word truth and leave the builder untouched, on
  ``segments[].words`` and its concatenation ``word_segments``.
- Summarises each segment with the duration-weighted majority speaker of its
  own words; a segment is '-1' only when every one of its words is.
- Carries each backend's real per-word start, end and confidence through to the
  response instead of interpolating them from the segment span.
- Flags words and segments whose span contains concurrently active speakers.
- Clamps adjacent segment overlaps to guarantee non-overlapping output.
- Builds transcript, diarization, and raw_words convenience fields.

See ADR 0014 for why segmentation is sentence-first and the segment label is a
summary rather than a guarantee.
"""

from __future__ import annotations

from coro.core.models import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptToken,
    TranscriptWord,
)
from coro.core.segmentation import group_tokens_into_runs
from coro.core.speakers import (
    UNKNOWN_SPEAKER,
    SpeakerAttribution,
    attribute_span,
    merge_speaker_timeline,
)

_AttributedToken = tuple[TranscriptToken, SpeakerAttribution]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _word_from_token(
    token: TranscriptToken,
    attribution: SpeakerAttribution,
) -> TranscriptWord:
    """Build a response word from a token's *real* timing, confidence and speaker.

    The word keeps its *own* attribution. Nothing here is overwritten by the
    segment's label: ``word_segments`` is the normative per-word speaker truth
    and the segment label is a summary of it, not the other way round.

    An absent probability stays absent. Substituting ``1.0`` would publish the
    strongest possible certainty precisely where the backend expressed none, and
    it travels: it is averaged into utterance confidences and emitted under a
    vendor's ``confidence`` key, where a client cannot tell it from a measurement.
    """
    return TranscriptWord(
        word=token.text.strip(),
        start=round(token.start, 2),
        end=round(token.end, 2),
        score=float(token.probability) if token.probability is not None else None,
        speaker=str(attribution.speaker),
        overlap=attribution.overlap,
    )


def _majority_speaker(run: list[_AttributedToken]) -> int:
    """Return the duration-weighted majority speaker of a run's words.

    Each word votes with its own duration, so a few long words outweigh many
    short ones. :data:`~coro.core.speakers.UNKNOWN_SPEAKER` words *abstain* — they
    contribute no duration and are never counted — which is exactly what makes a
    segment ``-1`` only when every one of its words is. Letting abstentions vote
    would let a diarization gap outweigh a speaker the diarizer did cover.

    Ties break on total word count, then on the lowest speaker label. The last
    key mirrors :func:`coro.core.speakers.attribute_span`, so the whole pipeline
    is order-independent, and it also makes the degenerate all-zero-duration case
    well defined instead of dictionary-order dependent.
    """
    duration: dict[int, float] = {}
    count: dict[int, int] = {}
    for token, attribution in run:
        speaker = attribution.speaker
        if speaker == UNKNOWN_SPEAKER:
            continue
        duration[speaker] = duration.get(speaker, 0.0) + token.duration()
        count[speaker] = count.get(speaker, 0) + 1
    if not duration:
        return UNKNOWN_SPEAKER
    # sorted() + max()'s first-maximum semantics make the final tie-break the
    # lowest speaker label, exactly as attribute_span does it.
    return max(sorted(duration), key=lambda label: (duration[label], count[label]))


def _segment_from_run(run: list[_AttributedToken]) -> ResponseSegment:
    """Collapse a sentence-shaped run of attributed tokens into one segment."""
    words = [_word_from_token(token, attribution) for token, attribution in run]
    start = min(token.start for token, _ in run)
    end = max(token.end for token, _ in run)
    if end < start:
        start, end = end, start
    return ResponseSegment(
        start=round(start, 2),
        end=round(end, 2),
        text="".join(token.text for token, _ in run).strip(),
        speaker=str(_majority_speaker(run)),
        words=words,
        overlap=any(word.overlap for word in words),
    )


def _clamp_overlaps(segments: list[ResponseSegment]) -> list[ResponseSegment]:
    """Clamp adjacent segment end times to eliminate overlapping ranges."""
    ordered = sorted(segments, key=lambda s: s.start)
    for i in range(len(ordered) - 1):
        current = ordered[i]
        nxt = ordered[i + 1]
        if current.end > nxt.start:
            current.end = round(max(current.start, nxt.start), 2)
    return ordered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_response_segment(
    tokens: list[TranscriptToken],
    merged_timeline: list[SpeakerSegment],
) -> ResponseSegment | None:
    """Turn one segment run into one response segment.

    Every token is attributed independently and keeps its own label. The run is
    *not* cut where the word-level speaker changes: boundaries stay
    sentence-shaped and come from the transcript alone. The segment's own
    ``speaker`` is the duration-weighted majority of its words, an auditable
    summary sitting beside the per-word truth in ``word_segments`` rather than a
    homogeneity guarantee.

    Args:
        tokens: One segment run from :mod:`coro.core.segmentation`.
        merged_timeline: Output of
            :func:`coro.core.speakers.merge_speaker_timeline`.

    Returns:
        One segment, or ``None`` when the run carries no transcript. Shared by
        the batch builder and the Streaming Pipeline's assembly so both emit
        identical segments.

    """
    attributed: list[_AttributedToken] = [
        (token, attribute_span(token.start, token.end, merged_timeline))
        for token in tokens
        if token.text and token.text.strip()
    ]
    if not attributed:
        return None
    return _segment_from_run(attributed)


def build_transcription_response(
    tokens: list[TranscriptToken],
    speaker_timeline: list[SpeakerSegment],
    duration: float,
) -> TranscriptionResult:
    """Build a :class:`TranscriptionResult` from project-owned types.

    Args:
        tokens: Ordered transcript tokens (Project-Owned Transcript Model).
        speaker_timeline: Speaker timeline segments from the Diarization Adapter.
        duration: Total audio duration in seconds.

    Returns:
        TranscriptionResult with segments, word_segments, transcript,
        diarization, and raw_words.

    """
    if not tokens:
        diar = [
            DiarizationItem(start=round(s.start, 3), end=round(s.end, 3), speaker=str(s.speaker))
            for s in sorted(speaker_timeline, key=lambda x: x.start)
        ]
        return TranscriptionResult(diarization=diar)

    raw_words = [
        RawWord(
            word=t.text,
            start=round(t.start, 3),
            end=round(t.end, 3),
            score=float(t.probability) if t.probability is not None else 1.0,
        )
        for t in tokens
        if t.text and t.text.strip()
    ]

    merged = merge_speaker_timeline(speaker_timeline)
    built: list[ResponseSegment] = []
    for run in group_tokens_into_runs(tokens):
        segment = build_response_segment(run, merged)
        if segment is not None:
            built.append(segment)
    segments = _clamp_overlaps(built)

    word_segments: list[TranscriptWord] = []
    for segment in segments:
        word_segments.extend(segment.words)

    transcript = [TranscriptItem(start=s.start, end=s.end, text=s.text) for s in segments]
    diarization = [DiarizationItem(start=s.start, end=s.end, speaker=s.speaker) for s in segments]

    return TranscriptionResult(
        segments=segments,
        word_segments=word_segments,
        transcript=transcript,
        diarization=diarization,
        raw_words=raw_words,
    )
