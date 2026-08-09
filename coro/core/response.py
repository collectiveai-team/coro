"""Core response builder.

Accepts Project-Owned Transcript Model types and produces the enriched
:class:`TranscriptionResult`.  No FastAPI or backend-native types are used.

Key behaviours:
- Groups tokens into segment runs using the Spanish-aware policy in
  :mod:`coro.core.segmentation`.
- Assigns a speaker to every *word* from the diarization timeline, absorbs
  sandwiched flicker islands via
  :func:`coro.core.realignment.realign_speaker_flicker`, then splits a run
  wherever the (realigned) word-level speaker changes.
- Carries each backend's real per-word start, end and confidence through to the
  response instead of interpolating them from the segment span.
- Flags words and segments whose span contains concurrently active speakers.
- Clamps adjacent segment overlaps to guarantee non-overlapping output.
- Emits speaker='-1' for spans the diarization timeline does not support.
- Builds transcript, diarization, and raw_words convenience fields.
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
from coro.core.realignment import realign_speaker_flicker
from coro.core.segmentation import group_tokens_into_runs
from coro.core.speakers import (
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
    speaker: str,
) -> TranscriptWord:
    """Build a response word from a token's *real* timing and confidence."""
    return TranscriptWord(
        word=token.text.strip(),
        start=round(token.start, 2),
        end=round(token.end, 2),
        score=float(token.probability) if token.probability is not None else 1.0,
        speaker=speaker,
        overlap=attribution.overlap,
    )


def _segment_from_run(run: list[_AttributedToken]) -> ResponseSegment:
    """Collapse a speaker-homogeneous run of attributed tokens into a segment."""
    speaker = str(run[0][1].speaker)
    words = [_word_from_token(token, attribution, speaker) for token, attribution in run]
    start = min(token.start for token, _ in run)
    end = max(token.end for token, _ in run)
    if end < start:
        start, end = end, start
    return ResponseSegment(
        start=round(start, 2),
        end=round(end, 2),
        text="".join(token.text for token, _ in run).strip(),
        speaker=speaker,
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


def build_response_segments(
    tokens: list[TranscriptToken],
    merged_timeline: list[SpeakerSegment],
) -> list[ResponseSegment]:
    """Turn one segment run into speaker-homogeneous response segments.

    Every token is attributed independently, then the run is split wherever the
    word-level speaker changes, so a run spanning a speaker turn no longer
    stamps one label onto every word after the turn.

    Args:
        tokens: One segment run from :mod:`coro.core.segmentation`.
        merged_timeline: Output of
            :func:`coro.core.speakers.merge_speaker_timeline`.

    Returns:
        Zero or more segments, in token order. Shared by the batch builder and
        the Streaming Pipeline's assembly so both emit identical segments.

    """
    attributed: list[_AttributedToken] = [
        (token, attribute_span(token.start, token.end, merged_timeline))
        for token in tokens
        if token.text and token.text.strip()
    ]
    if not attributed:
        return []
    attributed = realign_speaker_flicker(attributed)

    segments: list[ResponseSegment] = []
    run: list[_AttributedToken] = []
    for item in attributed:
        if run and item[1].speaker != run[0][1].speaker:
            segments.append(_segment_from_run(run))
            run = []
        run.append(item)
    if run:
        segments.append(_segment_from_run(run))
    return segments


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
        built.extend(build_response_segments(run, merged))
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
