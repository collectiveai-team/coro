"""Flat-memory streaming of the SSE ``transcript.text.done`` frame.

The done event carries the whole transcription response as a JSON string.
Materialising it would reintroduce an O(audio length) peak at end-of-stream,
defeating flat-memory streaming.  ``StreamingDoneFrame`` instead generates the
frame fragment-by-fragment, holding only one segment or word in memory at a
time.

The emitted bytes are byte-identical to the materialised path
(``TranscriptDoneEvent(text=json.dumps(build_streaming_response(...)))`` framed
by the SSE generator).  This holds because JSON string escaping is per
character, so escaping concatenated fragments equals escaping the whole.

Two properties make that equality hard to break by accident:

- **Nothing is hardcoded.**  The event envelope and the response field order are
  derived from ``TranscriptDoneEvent`` and ``TranscriptionResult`` at import
  time, so adding or reordering a field cannot silently desynchronise the
  streaming and batch outputs.  A response field with no renderer fails loudly
  at import rather than vanishing from the streamed frame.
- **One pass.**  The stored segments are scanned exactly once.  Each response
  array derived from them is spooled to its own disk-backed buffer during that
  pass and streamed back out in field order, so single-pass rendering costs no
  resident memory.
"""

from __future__ import annotations

import codecs
import json
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from coro.core.models import (
    DiarizationItem,
    ResponseSegment,
    SpeakerSegment,
    TranscriptDoneEvent,
    TranscriptionResult,
    TranscriptItem,
)
from coro.pipelines.finalizer import iter_response_segments
from coro.pipelines.transcript_store import TranscriptSpillStore

# json.dumps' default separators, which the materialised path uses.
_ELEMENT_SEPARATOR = ", "
_KEY_SEPARATOR = ": "

# Spool IO buffer sizes.  These, plus one escaped copy of a read chunk, are the
# frame's entire resident footprint, so they are the flat-memory budget and are
# deliberately small rather than throughput-optimal.
_SPOOL_BUFFER_BYTES = 4096
_SPOOL_READ_BYTES = 4096

# Improbable-in-a-transcript marker used only to locate the ``text`` slot in a
# serialised done event; never emitted.
_DONE_TEXT_SENTINEL = "\x00coro-done-frame-text\x00"


def _done_frame_affixes() -> tuple[str, str]:
    """Derive the SSE done-frame prefix and suffix from ``TranscriptDoneEvent``.

    Serialises a sentinel-valued event exactly as the SSE generator would, then
    splits on the sentinel.  The envelope therefore tracks the dataclass instead
    of being mirrored by hand.
    """
    envelope = json.dumps(asdict(TranscriptDoneEvent(text=_DONE_TEXT_SENTINEL)))
    escaped_sentinel = json.dumps(_DONE_TEXT_SENTINEL)[1:-1]
    prefix, separator, suffix = envelope.partition(escaped_sentinel)
    if not separator:  # pragma: no cover - defensive
        raise RuntimeError("Could not locate the done-event text slot in its JSON envelope.")
    return f"data: {prefix}", f"{suffix}\n\n"


_FRAME_PREFIX, _FRAME_SUFFIX = _done_frame_affixes()


# MARK: Response Array Renderers
def _segment_elements(segment: ResponseSegment) -> Iterator[str]:
    """Yield the ``segments`` element for one stored segment."""
    yield json.dumps(asdict(segment))


def _word_segment_elements(segment: ResponseSegment) -> Iterator[str]:
    """Yield the ``word_segments`` elements contributed by one stored segment."""
    for word in segment.words:
        yield json.dumps(asdict(word))


def _segment_projection(item_type: type) -> Callable[[ResponseSegment], Iterator[str]]:
    """Build a renderer projecting a stored segment onto a response item type.

    The convenience arrays (``transcript``, ``diarization``) are views over the
    stored segment: every field of the item type names an attribute of
    :class:`ResponseSegment`.  Reading the field names off the item dataclass
    keeps key order and key names in sync with it without constructing an
    intermediate object per segment.
    """
    field_names = tuple(f.name for f in fields(item_type))
    segment_fields = {f.name for f in fields(ResponseSegment)}
    missing = [name for name in field_names if name not in segment_fields]
    if missing:  # pragma: no cover - import-time contract guard
        raise RuntimeError(
            f"{item_type.__name__} fields {missing} have no ResponseSegment "
            "counterpart, so the streamed done frame cannot project them."
        )

    def render(segment: ResponseSegment) -> Iterator[str]:
        yield json.dumps({name: getattr(segment, name) for name in field_names})

    return render


_transcript_elements = _segment_projection(TranscriptItem)
_diarization_elements = _segment_projection(DiarizationItem)


def _raw_word_elements(store: TranscriptSpillStore) -> Iterator[str]:
    """Yield the ``raw_words`` elements straight from the store's own table."""
    for word in store.iter_raw_words():
        yield json.dumps(asdict(word))


# Response arrays derived from the stored segments, rendered in one shared pass.
_SEGMENT_DERIVED_RENDERERS = {
    "segments": _segment_elements,
    "word_segments": _word_segment_elements,
    "transcript": _transcript_elements,
    "diarization": _diarization_elements,
}

# Response arrays read directly from the store, needing no segment pass.
_STORE_DERIVED_RENDERERS = {"raw_words": _raw_word_elements}

# Field order comes from the response dataclass, never from string literals.
_RESPONSE_FIELDS = tuple(f.name for f in fields(TranscriptionResult))

_UNRENDERED = (
    set(_RESPONSE_FIELDS) - set(_SEGMENT_DERIVED_RENDERERS) - set(_STORE_DERIVED_RENDERERS)
)
_UNKNOWN = (set(_SEGMENT_DERIVED_RENDERERS) | set(_STORE_DERIVED_RENDERERS)) - set(_RESPONSE_FIELDS)
if _UNRENDERED or _UNKNOWN:  # pragma: no cover - import-time contract guard
    raise RuntimeError(
        "StreamingDoneFrame renderers are out of sync with TranscriptionResult: "
        f"unrendered fields {sorted(_UNRENDERED)}, unknown renderers {sorted(_UNKNOWN)}."
    )


# MARK: Fragment Spool
class _ArraySpool:
    """A disk-backed, append-only buffer holding one JSON array's elements.

    Separators are written as elements arrive, so reading back is a plain byte
    stream.  The file is unlinked on creation, so an abandoned frame leaks
    nothing.  Binary IO with an explicit incremental decoder avoids a text
    wrapper's extra buffers while still never splitting a code point across
    fragments, which per-character JSON escaping requires.
    """

    def __init__(self, directory: str | None) -> None:
        # Not a `with` block: the spool outlives this call and owns the file for
        # its lifetime, releasing it in close() and in read_fragments().
        self._file = tempfile.TemporaryFile(  # noqa: SIM115
            mode="w+b", buffering=_SPOOL_BUFFER_BYTES, dir=directory
        )
        self._count = 0

    def append(self, element: str) -> None:
        """Append one already-serialised array element."""
        if self._count:
            self._file.write(_ELEMENT_SEPARATOR.encode("utf-8"))
        self._file.write(element.encode("utf-8"))
        self._count += 1

    def read_fragments(self) -> Iterator[str]:
        """Yield the spooled array body in bounded chunks, then release the file."""
        try:
            self._file.flush()
            self._file.seek(0)
            decoder = codecs.getincrementaldecoder("utf-8")()
            while chunk := self._file.read(_SPOOL_READ_BYTES):
                text = decoder.decode(chunk)
                if text:
                    yield text
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail
        finally:
            self.close()

    def close(self) -> None:
        """Release the spool file."""
        self._file.close()


def _escape(fragment: str) -> str:
    """Escape a raw fragment for embedding inside a JSON string literal."""
    return json.dumps(fragment)[1:-1]


def _separated(elements: Iterator[str]) -> Iterator[str]:
    """Yield array elements interleaved with JSON's element separator."""
    for i, element in enumerate(elements):
        if i:
            yield _ELEMENT_SEPARATOR
        yield element


# MARK: Done Frame
@dataclass
class StreamingDoneFrame:
    """A store-backed, lazily rendered SSE done frame."""

    store: TranscriptSpillStore
    timeline: list[SpeakerSegment]

    def inner_fragments(self) -> Iterator[str]:
        """Yield the response JSON piecewise, equal to json.dumps(response)."""
        spools = {name: _ArraySpool(self._spool_directory()) for name in _SEGMENT_DERIVED_RENDERERS}
        try:
            self._spool_segment_arrays(spools)
            yield "{"
            for i, name in enumerate(_RESPONSE_FIELDS):
                if i:
                    yield _ELEMENT_SEPARATOR
                yield f"{json.dumps(name)}{_KEY_SEPARATOR}["
                if name in spools:
                    yield from spools[name].read_fragments()
                else:
                    yield from _separated(_STORE_DERIVED_RENDERERS[name](self.store))
                yield "]"
            yield "}"
        finally:
            for spool in spools.values():
                spool.close()

    def iter_sse(self) -> Iterator[str]:
        """Yield the complete SSE frame, escaped, ready to write to the client.

        The frame owns the spill store: it is closed once the frame has been
        fully rendered (or the generator is abandoned), so the store outlives
        the producing pipeline regardless of when the consumer renders it.
        """
        try:
            yield _FRAME_PREFIX
            for fragment in self.inner_fragments():
                yield _escape(fragment)
            yield _FRAME_SUFFIX
        finally:
            self.store.close()

    def _spool_directory(self) -> str:
        """Return the store's own directory, already validated as real disk."""
        return str(Path(self.store.path).parent)

    def _spool_segment_arrays(self, spools: dict[str, _ArraySpool]) -> None:
        """Fill every segment-derived spool in a single pass over the store."""
        for segment in iter_response_segments(self.store, self.timeline):
            for name, render in _SEGMENT_DERIVED_RENDERERS.items():
                spool = spools[name]
                for element in render(segment):
                    spool.append(element)
