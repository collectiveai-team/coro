"""Transcript Source — the read interface every response projection renders from.

A vendor projection needs the five response arrays and two scalars derived from
them. Taking those as *iterators* rather than as a ``TranscriptionResult``
decouples the projection from whether the transcript is resident: the
**Full-Memory Pipeline** hands over lists it already holds, the **Streaming
Pipeline** hands over a cursor into the **Transcript Spill Store**, and one
implementation of each response format serves both. Byte parity between the two
pipelines stops depending on two projections being kept in step (ADR 0018).

The five iterators mirror ``TranscriptionResult``'s five fields exactly, rather
than deriving four of them from ``segments``, because the derivation is not
total: an empty transcript with a speaker timeline yields a populated
``diarization`` and no segments at all. A source that recomputed the convenience
arrays would silently change that case.

A source may be iterated several times — once per array it feeds, plus once for
the duration — so every iterator must be restartable and none may consume state.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from coro.core.models import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)


class TranscriptSource(Protocol):
    """Restartable, order-preserving reads over one transcription's response arrays."""

    def iter_segments(self) -> Iterator[ResponseSegment]:
        """Yield speaker-attributed, overlap-clamped segments in transcript order."""
        ...

    def iter_words(self) -> Iterator[TranscriptWord]:
        """Yield the per-word view in transcript order."""
        ...

    def iter_transcript(self) -> Iterator[TranscriptItem]:
        """Yield the transcript convenience view in transcript order."""
        ...

    def iter_diarization(self) -> Iterator[DiarizationItem]:
        """Yield the diarization convenience view in transcript order."""
        ...

    def iter_raw_words(self) -> Iterator[RawWord]:
        """Yield raw ASR words as the backend emitted them."""
        ...

    def close(self) -> None:
        """Release whatever backs the source."""
        ...


class MemoryTranscriptSource:
    """A Transcript Source over an already-materialised result.

    Used by the **Full-Memory Pipeline**, which holds the whole decoded PCM
    regardless and so has nothing to gain from a lazy transcript. It exists so
    that pipeline reaches the wire through the *same* projection code, not to
    make it flat.
    """

    def __init__(self, result: TranscriptionResult) -> None:
        self._result = result

    @property
    def detected_language(self) -> str | None:
        """Language auto-LID resolved for this request, if any.

        Not part of the :class:`TranscriptSource` Protocol (a lazy source
        without one simply has no attribute; callers read it with
        ``getattr(source, "detected_language", None)``, see
        ``coro/api/openai/render.py``).
        """
        return self._result.detected_language

    def iter_segments(self) -> Iterator[ResponseSegment]:
        return iter(self._result.segments)

    def iter_words(self) -> Iterator[TranscriptWord]:
        return iter(self._result.word_segments)

    def iter_transcript(self) -> Iterator[TranscriptItem]:
        return iter(self._result.transcript)

    def iter_diarization(self) -> Iterator[DiarizationItem]:
        return iter(self._result.diarization)

    def iter_raw_words(self) -> Iterator[RawWord]:
        return iter(self._result.raw_words)

    def close(self) -> None:
        return None


def response_duration(source: TranscriptSource) -> float:
    """Return the response's reported duration: the latest end across every array.

    Mirrors the materialised projection's own definition exactly, including its
    ``default=0.0`` for a response with no timed content. It is a separate pass
    because every format serialises a duration *before* the arrays it summarises,
    so it cannot be accumulated while rendering them.
    """
    return max(
        (
            item.end
            for items in (
                source.iter_segments(),
                source.iter_words(),
                source.iter_raw_words(),
                source.iter_transcript(),
                source.iter_diarization(),
            )
            for item in items
        ),
        default=0.0,
    )


def iter_text_fragments(source: TranscriptSource) -> Iterator[str]:
    """Yield the full transcript text piecewise, equal to the materialised join.

    The materialised form is ``" ".join(item.text.strip() for item in ...).strip()``.
    Reproducing that incrementally needs care: the outer ``strip()`` removes
    whitespace only at the *ends*, so the runs of spaces that empty items
    contribute in the *middle* must be preserved exactly. Separators are
    therefore withheld until a non-empty item actually follows one, and dropped
    entirely before the first and after the last.
    """
    texts = (item.text for item in _texts_source(source))

    pending_separators = 0
    emitted = False
    for index, raw in enumerate(texts):
        if index:
            pending_separators += 1
        text = raw.strip()
        if not text:
            continue
        if emitted:
            yield " " * pending_separators
        pending_separators = 0
        emitted = True
        yield text


def _texts_source(source: TranscriptSource) -> Iterator[TranscriptItem] | Iterator[ResponseSegment]:
    """Return the array the transcript text is read from.

    The materialised projection prefers ``transcript`` and falls back to
    ``segments``; the two carry identical text whenever both are populated, so
    the fallback only matters for a result that has segments but no transcript.
    Peeking one item is what makes "is it empty" answerable without materialising
    either.
    """
    transcript = source.iter_transcript()
    first = next(transcript, None)
    if first is not None:
        return _prepend(first, transcript)
    return source.iter_segments()


def _prepend[T](first: T, rest: Iterator[T]) -> Iterator[T]:
    """Put a peeked item back in front of its iterator."""
    yield first
    yield from rest
