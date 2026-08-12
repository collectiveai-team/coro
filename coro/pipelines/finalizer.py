"""Incremental transcript finalization for flat-memory streaming.

The batch response builder groups *all* tokens into segment runs at once, which
requires retaining the entire transcript.  Because tokens arrive in order and
are never reordered, a run is final the instant its closing token arrives:
nothing later inserts before it.  This finalizer exploits that to spill
finalized runs and raw words to a :class:`TranscriptSpillStore`, keeping only
the current open run of tokens in memory.

Runs are spilled as *tokens*, not as assembled segments: the streaming diarizer
only produces its complete timeline once the audio ends, so word-level speaker
attribution — and the segment's majority label that summarises it — is deferred
to assembly (:func:`build_streaming_response`).  Assembly then calls the very
same :func:`coro.core.response.build_response_segment` the batch builder uses,
one run at a time, so both paths are identical by construction while memory
stays flat.

Run boundaries are sentence-shaped and decided from the transcript alone, so
they are final at spill time and do not depend on the diarization timeline that
has not arrived yet (ADR 0014).
"""

from __future__ import annotations

from collections.abc import Iterator

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
from coro.core.response import build_response_segment
from coro.core.segmentation import SegmentAccumulator, run_span
from coro.core.speakers import merge_speaker_timeline
from coro.pipelines.transcript_store import TranscriptSpillStore


class StreamingTranscriptFinalizer:
    """Group tokens into finalized segment runs and spill them to a store."""

    def __init__(self, store: TranscriptSpillStore) -> None:
        self._store = store
        self._accumulator = SegmentAccumulator()

    @property
    def open_tokens(self) -> list[TranscriptToken]:
        """Tokens of the current unterminated run (bounded working state)."""
        return self._accumulator.open_tokens

    def add_tokens(self, tokens: list[TranscriptToken]) -> None:
        """Ingest a batch of accepted tokens, finalizing completed runs."""
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
            for run in self._accumulator.add(token):
                self._spill(run)

    def finish(self) -> None:
        """Finalize the trailing open run (the unterminated final segment)."""
        self._spill(self._accumulator.flush())

    def _spill(self, run: list[TranscriptToken]) -> None:
        span = run_span(run)
        if span is None:
            return
        start, end, text = span
        self._store.append_segment_tokens(run, start=start, end=end, text=text)


def iter_response_segments(
    store: TranscriptSpillStore,
    speaker_timeline: list[SpeakerSegment] | None = None,
) -> Iterator[ResponseSegment]:
    """Yield finalized segments, speaker-attributed and overlap-clamped.

    Each stored run is attributed at word granularity and labelled with the
    duration-weighted majority of its own words, then a one-segment-lookahead
    clamp ensures a segment never ends past the next one's start (matching
    ``_clamp_overlaps`` for in-order input).  Only a single segment is buffered,
    so memory stays flat.
    """
    merged = merge_speaker_timeline(speaker_timeline or [])

    prev: ResponseSegment | None = None
    for tokens in store.iter_segment_tokens():
        seg = build_response_segment(tokens, merged)
        if seg is None:
            continue
        if prev is not None:
            if prev.end > seg.start:
                prev.end = round(max(prev.start, seg.start), 2)
            yield prev
        prev = seg
    if prev is not None:
        yield prev


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
