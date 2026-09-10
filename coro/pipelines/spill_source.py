"""A Transcript Source backed by the Transcript Spill Store.

The Streaming Pipeline finishes a request holding a transcript on disk and a
speaker timeline in memory. This turns that pair into the same read interface
the Full-Memory Pipeline offers over its lists, so one response projection
serves both (ADR 0018).

Every iterator re-queries the store rather than caching, which is what keeps
resident memory bounded and what makes the source restartable — a projection
walks it once per array it renders. The convenience arrays are derived from the
segments here exactly as ``build_streaming_response`` derives them, so
materialising this source reproduces that function's output field for field.
"""

from __future__ import annotations

from collections.abc import Iterator

from coro.core.models import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    SpeakerSegment,
    TranscriptItem,
    TranscriptWord,
)
from coro.pipelines.finalizer import iter_response_segments
from coro.pipelines.transcript_store import TranscriptSpillStore


class SpillTranscriptSource:
    """A Transcript Source reading through a per-request spill store."""

    def __init__(
        self,
        store: TranscriptSpillStore,
        timeline: list[SpeakerSegment],
        *,
        detected_language: str | None = None,
    ) -> None:
        self._store = store
        self._timeline = timeline
        self._detected_language = detected_language

    @property
    def detected_language(self) -> str | None:
        """Language auto-LID resolved for this request, if any.

        Not part of the :class:`~coro.core.transcript_source.TranscriptSource`
        Protocol; callers read it with ``getattr(source, "detected_language",
        None)`` (see ``coro/api/openai/render.py``), mirroring
        ``MemoryTranscriptSource``'s own property.
        """
        return self._detected_language

    def iter_segments(self) -> Iterator[ResponseSegment]:
        return iter_response_segments(self._store, self._timeline)

    def iter_words(self) -> Iterator[TranscriptWord]:
        for segment in self.iter_segments():
            yield from segment.words

    def iter_transcript(self) -> Iterator[TranscriptItem]:
        for segment in self.iter_segments():
            yield TranscriptItem(start=segment.start, end=segment.end, text=segment.text)

    def iter_diarization(self) -> Iterator[DiarizationItem]:
        for segment in self.iter_segments():
            yield DiarizationItem(start=segment.start, end=segment.end, speaker=segment.speaker)

    def iter_raw_words(self) -> Iterator[RawWord]:
        return self._store.iter_raw_words()

    def close(self) -> None:
        """Close the store, deleting the database and its WAL sidecars."""
        self._store.close()
