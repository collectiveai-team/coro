"""Incremental transcript finalization for flat-memory streaming.

The batch response builder groups *all* tokens into punctuation-boundary
segments at once, which requires retaining the entire transcript.  Because
tokens arrive in order and are never reordered, a segment is final the instant
its closing-punctuation token arrives: nothing later inserts before it.  This
finalizer exploits that to emit finalized segments incrementally, keeping only
the current open run of tokens in memory and spilling finalized segments and
raw words to a :class:`TranscriptSpillStore`.

Speaker labels are resolved per segment at finalization time via an injected
resolver (online assignment), so no global pass over the full transcript and
diarization timeline is needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from asr_diar_server.core.response import (
    build_segment_dict,
    closes_segment,
    segment_span_from_tokens,
)
from asr_diar_server.core.types import TranscriptSegment, TranscriptToken
from asr_diar_server.pipelines.transcript_store import TranscriptSpillStore

# Resolve a speaker label for a finalized ``[start, end)`` segment span.
SpeakerResolver = Callable[[float, float], int]


def _default_speaker(start: float, end: float) -> int:
    """Default resolver: speaker 1, matching the no-diarization batch default."""
    return 1


class StreamingTranscriptFinalizer:
    """Group tokens into finalized segments and spill them to a store."""

    def __init__(
        self,
        store: TranscriptSpillStore,
        *,
        speaker_resolver: SpeakerResolver | None = None,
    ) -> None:
        self._store = store
        self._resolve = speaker_resolver or _default_speaker
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
        speaker = self._resolve(start, end)
        seg = TranscriptSegment(start=start, end=end, text=text, speaker=speaker)
        self._store.append_segment(build_segment_dict(seg))


def iter_response_segments(store: TranscriptSpillStore) -> Iterator[dict]:
    """Yield finalized segments with a one-segment-lookahead overlap clamp.

    Adjacent segments are clamped so a segment never ends past the next one's
    start, matching the batch builder's ``_clamp_overlaps`` for in-order input
    while buffering only a single segment (flat memory).
    """
    prev: dict | None = None
    for seg in store.iter_segments():
        if prev is not None:
            if prev["end"] > seg["start"]:
                prev["end"] = round(max(prev["start"], seg["start"]), 2)
            yield prev
        prev = seg
    if prev is not None:
        yield prev


def build_streaming_response(store: TranscriptSpillStore) -> dict:
    """Assemble the full response dict from a spill store.

    Mirrors the keys of the batch builder.  This materialises the lists once
    (inherent for a single response object); steady-state streaming stays flat
    because the data lived in the store, not Python lists.
    """
    segments: list[dict] = []
    word_segments: list[dict] = []
    for seg in iter_response_segments(store):
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
