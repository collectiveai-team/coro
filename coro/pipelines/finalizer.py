"""Incremental transcript finalization for flat-memory streaming.

The batch response builder groups *all* tokens into punctuation-boundary
segments at once, which requires retaining the entire transcript.  Because
tokens arrive in order and are never reordered, a segment is final the instant
its closing-punctuation token arrives: nothing later inserts before it.  This
finalizer exploits that to emit finalized segments incrementally, keeping only
the current open run of tokens in memory and spilling finalized segments and
raw words to a :class:`TranscriptSpillStore`.

Speaker labels are NOT assigned here: the streaming diarizer only produces its
complete timeline once the audio ends, so attribution is deferred to assembly
(:func:`build_streaming_response`), which runs the same global max-overlap pass
as the batch builder in a flat, one-segment-at-a-time sweep over the store.
"""

from __future__ import annotations

from collections.abc import Iterator

from coro.core.response import (
    build_segment_dict,
    closes_segment,
    merge_speaker_timeline,
    segment_span_from_tokens,
    speaker_for_span,
)
from coro.core.types import SpeakerSegment, TranscriptSegment, TranscriptToken
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
                {
                    "word": t.text,
                    "start": round(t.start, 3),
                    "end": round(t.end, 3),
                    "score": float(t.probability) if t.probability is not None else 1.0,
                }
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
        # Provisional speaker 1; real attribution happens at assembly once the
        # diarizer has produced its complete timeline.
        seg = TranscriptSegment(start=start, end=end, text=text, speaker=1)
        self._store.append_segment(build_segment_dict(seg))


def _assign_segment_speaker(
    seg: dict,
    merged: list[SpeakerSegment],
    last_end: float,
    has_timeline: bool,
) -> None:
    """Assign a segment's (and its words') speaker from the merged timeline.

    Uses the segment's stored (pre-clamp) end so assignment matches the batch
    builder's order of assign-then-clamp.  With no timeline every segment
    defaults to speaker 1, mirroring ``_assign_speakers``.
    """
    spk = speaker_for_span(seg["start"], seg["end"], merged, last_end) if has_timeline else 1
    seg["speaker"] = str(spk)
    for word in seg["words"]:
        word["speaker"] = str(spk)


def iter_response_segments(
    store: TranscriptSpillStore,
    speaker_timeline: list[SpeakerSegment] | None = None,
) -> Iterator[dict]:
    """Yield finalized segments, speaker-attributed and overlap-clamped.

    Speakers are assigned per segment from the (complete) diarization timeline,
    then a one-segment-lookahead clamp ensures a segment never ends past the
    next one's start (matching ``_clamp_overlaps`` for in-order input).  Only a
    single segment is buffered, so memory stays flat.
    """
    merged = merge_speaker_timeline(speaker_timeline or [])
    last_end = merged[-1].end if merged else 0.0
    has_timeline = bool(merged)

    prev: dict | None = None
    for seg in store.iter_segments():
        _assign_segment_speaker(seg, merged, last_end, has_timeline)
        if prev is not None:
            if prev["end"] > seg["start"]:
                prev["end"] = round(max(prev["start"], seg["start"]), 2)
            yield prev
        prev = seg
    if prev is not None:
        yield prev


def build_streaming_response(
    store: TranscriptSpillStore,
    speaker_timeline: list[SpeakerSegment] | None = None,
) -> dict:
    """Assemble the full response dict from a spill store.

    Mirrors the keys of the batch builder.  This materialises the lists once
    (inherent for a single response object); steady-state streaming stays flat
    because the data lived in the store, not Python lists.
    """
    segments: list[dict] = []
    word_segments: list[dict] = []
    for seg in iter_response_segments(store, speaker_timeline):
        segments.append(seg)
        word_segments.extend(seg["words"])
    transcript = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]
    diarization = [
        {"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in segments
    ]
    raw_words = list(store.iter_raw_words())
    return {
        "segments": segments,
        "word_segments": word_segments,
        "transcript": transcript,
        "diarization": diarization,
        "raw_words": raw_words,
    }
