"""Incremental transcript finalization for flat-memory streaming.

The batch response builder groups *all* tokens into punctuation-boundary
segments at once, which requires retaining the entire transcript.  Because
tokens arrive in order and are never reordered, a segment is final the instant
its closing-punctuation token arrives: nothing later inserts before it.  This
finalizer exploits that to emit finalized segments incrementally, keeping only
the current open run of tokens in memory and spilling finalized segments and
raw words to a :class:`TranscriptSpillStore`.

Speaker labels, overlap clamping and word interpolation are NOT applied here.
Each depends on information that does not exist when a segment finalizes — the
streaming diarizer only produces its complete timeline once the audio ends, and
a segment's clamped end depends on the *next* segment's start.  All three are
deferred to assembly (:func:`iter_response_segments`), which applies them in
the batch builder's order — assign, clamp, then build words from the clamped
span — in a flat, one-segment-at-a-time sweep over the store.
"""

from __future__ import annotations

from collections.abc import Iterator

from coro.core.response import (
    build_segment,
    closes_segment,
    merge_speaker_timeline,
    segment_span_from_tokens,
    speaker_for_span,
)
from coro.core.models import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptSegment,
    TranscriptToken,
    TranscriptWord,
)
from coro.pipelines.transcript_store import TranscriptSpillStore


class StreamingTranscriptFinalizer:
    """Group tokens into finalized segments and spill them to a store."""

    def __init__(self, store: TranscriptSpillStore) -> None:
        self._store = store
        self._open: list[TranscriptToken] = []

    def add_tokens(self, tokens: list[TranscriptToken]) -> None:
        """Ingest a batch of accepted tokens, finalizing completed segments."""
        self._store.append_raw_words(
            [
                RawWord(
                    word=t.text,
                    start=round(t.start, 3),
                    end=round(t.end, 3),
                    score=float(t.probability) if t.probability is not None else 1.0,
                )
                for t in tokens
                if t.text and t.text.strip()
            ]
        )
        for token in tokens:
            if not token.text:
                continue
            self._open.append(token)
            if closes_segment(token.text):
                self._flush()

    def finish(self) -> None:
        """Finalize the trailing open run (the unterminated final segment)."""
        self._flush()

    def _flush(self) -> None:
        span = segment_span_from_tokens(self._open)
        self._open = []
        if span is None:
            return
        start, end, text = span
        self._store.append_segment(TranscriptSegment(start=start, end=end, text=text))


def iter_response_segments(
    store: TranscriptSpillStore,
    speaker_timeline: list[SpeakerSegment] | None = None,
) -> Iterator[ResponseSegment]:
    """Yield finalized segments, speaker-attributed, overlap-clamped and worded.

    Applies the batch builder's three steps in its order: assign the speaker
    from the (complete) diarization timeline using the raw span, clamp the end
    against the next segment's start (matching ``_clamp_overlaps`` for in-order
    input), then interpolate words over the clamped span via
    :func:`build_segment`.  Only one segment is buffered, so memory stays flat.
    """
    merged = merge_speaker_timeline(speaker_timeline or [])
    last_end = merged[-1].end if merged else 0.0
    has_timeline = bool(merged)

    prev: TranscriptSegment | None = None
    for seg in store.iter_segments():
        seg.speaker = speaker_for_span(seg.start, seg.end, merged, last_end) if has_timeline else 1
        if prev is not None:
            if prev.end > seg.start:
                prev.end = max(prev.start, seg.start)
            yield build_segment(prev)
        prev = seg
    if prev is not None:
        yield build_segment(prev)


def build_streaming_response(
    store: TranscriptSpillStore,
    speaker_timeline: list[SpeakerSegment] | None = None,
) -> TranscriptionResult:
    """Assemble the full :class:`TranscriptionResult` from a spill store.

    Mirrors the batch builder.  This materialises the lists once (inherent for
    a single response object); steady-state streaming stays flat because the
    data lived in the store, not Python lists.
    """
    segments: list[ResponseSegment] = []
    word_segments: list[TranscriptWord] = []
    for seg in iter_response_segments(store, speaker_timeline):
        segments.append(seg)
        word_segments.extend(seg.words)
    transcript = [TranscriptItem(start=s.start, end=s.end, text=s.text) for s in segments]
    diarization = [DiarizationItem(start=s.start, end=s.end, speaker=s.speaker) for s in segments]
    raw_words = list(store.iter_raw_words())
    return TranscriptionResult(
        segments=segments,
        word_segments=word_segments,
        transcript=transcript,
        diarization=diarization,
        raw_words=raw_words,
    )
