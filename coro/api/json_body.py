"""Incremental JSON response bodies that never fully materialise.

A vendor response is a small envelope of scalars wrapped around a few arrays
whose length is proportional to the audio. This renders the envelope from the
response model itself and splices the arrays in one element at a time, so the
resident cost is one element rather than the whole body (ADR 0018).

Two properties keep the streamed bytes identical to the materialised ones:

- **The envelope is derived, never written by hand.** Each renderer builds its
  real response model with *empty* arrays and a sentinel in place of the
  transcript text, serialises it, and splices into the resulting string. Key
  order, key names, separators and scalar formatting therefore come from the
  model, so adding or reordering a field cannot desynchronise the two paths. A
  slot whose marker is missing fails loudly instead of vanishing.
- **One serialiser.** ``dumps`` reproduces Starlette's ``JSONResponse`` exactly,
  which is what both vendor routes emit today. That is not interchangeable with
  Pydantic's own serialiser: ``model_dump_json`` writes ``1e-7`` where
  ``json.dumps`` writes ``1e-07``, so rendering elements with the wrong one
  would differ on some floats and agree on most, which is the worst way for a
  byte-identity guarantee to fail.

The rendered body is spooled to disk and served with a real ``Content-Length``,
so the HTTP framing is unchanged as well as the bytes.
"""

from __future__ import annotations

import codecs
import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Spool IO buffer sizes. These, plus one rendered element, are the body's entire
# resident footprint, so they are the flat-memory budget and are deliberately
# small rather than throughput-optimal.
_SPOOL_BUFFER_BYTES = 4096
_SPOOL_READ_BYTES = 8192

# Improbable-in-a-transcript marker used only to locate the text slot in a
# serialised envelope; never emitted.
TEXT_SENTINEL = "\x00coro-response-text\x00"

_ELEMENT_SEPARATOR = ","


def dumps(value: Any) -> str:
    """Serialise exactly as Starlette's ``JSONResponse`` does.

    Every keyword matters: the compact separators, the unescaped non-ASCII and
    the refusal of ``NaN``/``Infinity`` are all part of the bytes the vendor
    routes already emit.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def dump_model(model: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialise one response model the way its route would."""
    return dumps(model.model_dump(mode="json", exclude_none=exclude_none))


def escape_fragment(fragment: str) -> str:
    """Escape a raw fragment for embedding inside a JSON string literal.

    Escaping fragments separately equals escaping their concatenation because
    JSON string escaping is per character, and a ``str`` fragment never splits a
    code point.
    """
    return dumps(fragment)[1:-1]


# MARK: Envelope Slots
@dataclass(frozen=True)
class Slot:
    """A region of a serialised envelope replaced by streamed content."""

    marker: str
    opening: str
    closing: str
    fragments: Iterator[str]


def array_slot(name: str, elements: Iterator[str]) -> Slot:
    """Build a slot replacing an empty JSON array with its streamed elements."""
    key = f"{dumps(name)}:"
    return Slot(
        marker=f"{key}[]",
        opening=f"{key}[",
        closing="]",
        fragments=_separated(elements),
    )


def text_slot(fragments: Iterator[str]) -> Slot:
    """Build a slot replacing the sentinel string with streamed, escaped text."""
    return Slot(
        marker=dumps(TEXT_SENTINEL),
        opening='"',
        closing='"',
        fragments=(escape_fragment(fragment) for fragment in fragments),
    )


def _separated(elements: Iterator[str]) -> Iterator[str]:
    """Yield array elements interleaved with JSON's element separator."""
    for index, element in enumerate(elements):
        if index:
            yield _ELEMENT_SEPARATOR
        yield element


def splice(envelope: str, slots: list[Slot]) -> Iterator[str]:
    """Yield the envelope with each slot's marker replaced by streamed content.

    Slots must appear in serialisation order, which they do because each
    renderer lists them in the order its response model declares the fields.
    """
    remainder = envelope
    for slot in slots:
        prefix, separator, remainder = remainder.partition(slot.marker)
        if not separator:
            raise RuntimeError(
                f"Could not locate {slot.marker!r} in the serialised response envelope; "
                "the renderer is out of sync with its response model."
            )
        yield prefix
        yield slot.opening
        yield from slot.fragments
        yield slot.closing
    yield remainder


# MARK: Body Spool
class BodySpool:
    """A disk-backed buffer holding one rendered response body.

    Written fragment by fragment, then served back in bounded chunks with a real
    ``Content-Length`` taken from the file's size. The file is unlinked on
    creation, so an abandoned body leaks nothing.
    """

    def __init__(self, directory: str | None = None) -> None:
        # Not a `with` block: the spool outlives this call and owns the file
        # until the response has been streamed.
        self._file = tempfile.TemporaryFile(  # noqa: SIM115
            mode="w+b", buffering=_SPOOL_BUFFER_BYTES, dir=directory
        )
        self._size = 0

    def write(self, fragment: str) -> None:
        """Append one rendered fragment."""
        self._size += self._file.write(fragment.encode("utf-8"))

    @property
    def size(self) -> int:
        """Bytes written so far — the body's ``Content-Length``."""
        return self._size

    def iter_chunks(self) -> Iterator[bytes]:
        """Yield the spooled body in bounded chunks, then release the file."""
        try:
            self._file.flush()
            self._file.seek(0)
            while chunk := self._file.read(_SPOOL_READ_BYTES):
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        """Release the spool file."""
        self._file.close()


def spooled_json_response(
    fragments: Iterator[str],
    *,
    directory: str | None = None,
    status_code: int = 200,
) -> StreamingResponse:
    """Render fragments into a spool and return it as a length-declaring response.

    Rendering completes before the response is constructed, so a failure while
    projecting still surfaces as an ordinary exception the route's error mapping
    can convert — nothing has been written to the client yet.
    """
    spool = BodySpool(directory)
    try:
        for fragment in fragments:
            spool.write(fragment)
    except BaseException:
        spool.close()
        raise
    return StreamingResponse(
        spool.iter_chunks(),
        status_code=status_code,
        media_type="application/json",
        headers={"content-length": str(spool.size)},
    )


def iter_utf8(chunks: Iterator[bytes]) -> Iterator[str]:
    """Decode spooled chunks back to text without splitting a code point.

    Used by tests and by the offline command, which want the body as a string
    rather than as an HTTP response.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in chunks:
        text = decoder.decode(chunk)
        if text:
            yield text
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
