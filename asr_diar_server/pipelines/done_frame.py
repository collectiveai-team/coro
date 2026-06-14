"""Flat-memory streaming of the SSE ``transcript.text.done`` frame.

The done event carries the whole transcription response as a JSON string.
Materialising it would reintroduce an O(audio length) peak at end-of-stream,
defeating flat-memory streaming.  ``StreamingDoneFrame`` instead generates the
frame fragment-by-fragment straight from the spill store, holding only one
segment or word in memory at a time.

The emitted bytes are byte-identical to the materialised path
(``TranscriptDoneEvent(text=json.dumps(build_streaming_response(...)))`` framed
by the SSE generator).  This holds because JSON string escaping is per
character, so escaping concatenated fragments equals escaping the whole.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

from asr_diar_server.core.types import SpeakerSegment
from asr_diar_server.pipelines.finalizer import iter_response_segments
from asr_diar_server.pipelines.transcript_store import TranscriptSpillStore

# Done-frame envelope, matching json.dumps(asdict(TranscriptDoneEvent(...))).
_FRAME_PREFIX = 'data: {"text": "'
_FRAME_SUFFIX = '", "type": "transcript.text.done"}\n\n'


def _escape(fragment: str) -> str:
    """Escape a raw fragment for embedding inside a JSON string literal."""
    return json.dumps(fragment)[1:-1]


@dataclass
class StreamingDoneFrame:
    """A store-backed, lazily rendered SSE done frame."""

    store: TranscriptSpillStore
    timeline: list[SpeakerSegment]

    def inner_fragments(self) -> Iterator[str]:
        """Yield the response JSON piecewise, equal to json.dumps(response)."""
        yield '{"segments": ['
        for i, seg in enumerate(iter_response_segments(self.store, self.timeline)):
            yield ", " if i else ""
            yield json.dumps(seg)
        yield '], "word_segments": ['
        first = True
        for seg in iter_response_segments(self.store, self.timeline):
            for word in seg["words"]:
                yield "" if first else ", "
                first = False
                yield json.dumps(word)
        yield '], "transcript": ['
        for i, seg in enumerate(iter_response_segments(self.store, self.timeline)):
            yield ", " if i else ""
            yield json.dumps({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
        yield '], "diarization": ['
        for i, seg in enumerate(iter_response_segments(self.store, self.timeline)):
            yield ", " if i else ""
            yield json.dumps(
                {"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]}
            )
        yield '], "raw_words": ['
        for i, word in enumerate(self.store.iter_raw_words()):
            yield ", " if i else ""
            yield json.dumps(word)
        yield "]}"

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
