"""The Full-Memory and Streaming pipelines must emit byte-identical responses.

The Streaming Pipeline renders its SSE done frame straight from the spill
store instead of serialising a response object, so nothing in the type system
forces the two paths to agree.  Adding or reordering a field on
``TranscriptionResult`` — or on any item dataclass inside it — could silently
desynchronise batch and streaming output.

These tests pin the coupling at both levels: the assembled response JSON and
the fully framed SSE byte stream, for the same audio, the same ASR Windowing
configuration, and the same speaker timeline.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict
from unittest.mock import patch

import pytest

from coro.api.sse import _sse_generator
from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE, AudioInput
from coro.core.models import SpeakerSegment, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing

# 3.5 s of silence at a 1 s window with no overlap: three full windows plus a
# half-length tail, which both pipelines must window identically.
_AUDIO_SECONDS = 3.5
_PCM = struct.pack(f"<{int(SAMPLE_RATE * _AUDIO_SECONDS)}h", *([0] * int(SAMPLE_RATE * 3.5)))
_FEED_CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE // 2

# Two speakers spanning the audio, so segments exercise real attribution rather
# than the no-timeline default.
_TIMELINE = [
    SpeakerSegment(start=0.0, end=1.6, speaker=2),
    SpeakerSegment(start=1.6, end=3.5, speaker=3),
]

# Per-window token text. Quotes, non-ASCII and a trailing punctuation boundary
# exercise JSON escaping and segment finalization in both paths.
_WINDOW_TEXTS = (
    ' hola "mundo".',
    " ¿cómo estás?",
    " todo — bien.",
    " gracias!",
)


class _DeterministicASR:
    """Return window-indexed tokens so identical windowing yields identical tokens.

    Tokens deliberately end 0.2 s past their window so consecutive segments
    overlap, exercising each path's overlap clamp.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        text = _WINDOW_TEXTS[self.calls % len(_WINDOW_TEXTS)]
        self.calls += 1
        return [TranscriptToken(start=0.1, end=1.2, text=text, probability=0.75)]


class _BatchDiarizer:
    """Diarization Adapter returning the shared timeline in one call."""

    async def diarize_pcm(self, pcm):
        return list(_TIMELINE)


class _StreamingDiarizer:
    """Streaming Diarization Feed returning the same timeline at finalize."""

    def ingest_pcm_chunk(self, pcm: bytes) -> None:
        return None

    def finalize(self):
        return list(_TIMELINE)


async def _identity_pcm(data: bytes) -> bytes:
    """Stand in for ffmpeg decoding: the fixture is already PCM."""
    return data


async def _feed_pcm(path: str, chunk_seconds: float = 1.0):
    """Stand in for streamed ffmpeg decoding of the same fixture."""
    for offset in range(0, len(_PCM), _FEED_CHUNK_BYTES):
        yield _PCM[offset : offset + _FEED_CHUNK_BYTES]


def _windowing() -> ASRWindowing:
    return ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)


def _full_memory_pipeline(*, diarization: bool) -> FullMemoryPipeline:
    return FullMemoryPipeline(
        asr=_DeterministicASR(),
        diarization=_BatchDiarizer() if diarization else None,
        windowing=_windowing(),
    )


def _streaming_pipeline(spill_dir: str, *, diarization: bool) -> StreamingPipeline:
    return StreamingPipeline(
        asr=_DeterministicASR(),
        windowing=_windowing(),
        streaming_diarizer_factory=_StreamingDiarizer if diarization else None,
        spill_dir=spill_dir,
    )


async def _full_memory_response_json(*, diarization: bool) -> str:
    with patch("coro.pipelines.full_memory.convert_to_pcm_bytes", new=_identity_pcm):
        result = await _full_memory_pipeline(diarization=diarization).transcribe(AudioInput(_PCM))
    return json.dumps(asdict(result))


async def _streaming_response_json(spill_dir: str, *, diarization: bool) -> str:
    with patch("coro.pipelines.streaming.stream_pcm_from_file", new=_feed_pcm):
        result = await _streaming_pipeline(spill_dir, diarization=diarization).transcribe(
            AudioInput(_PCM)
        )
    return json.dumps(asdict(result))


async def _full_memory_sse(*, diarization: bool) -> str:
    pipeline = _full_memory_pipeline(diarization=diarization)
    with patch("coro.pipelines.full_memory.convert_to_pcm_bytes", new=_identity_pcm):
        return "".join([line async for line in _sse_generator(pipeline.stream(AudioInput(_PCM)))])


async def _streaming_sse(spill_dir: str, *, diarization: bool) -> str:
    pipeline = _streaming_pipeline(spill_dir, diarization=diarization)
    with patch("coro.pipelines.streaming.stream_pcm_from_file", new=_feed_pcm):
        return "".join([line async for line in _sse_generator(pipeline.stream(AudioInput(_PCM)))])


@pytest.mark.parametrize("diarization", [False, True])
@pytest.mark.asyncio
async def test_batch_response_json_is_byte_identical(tmp_path, diarization: bool):
    full_memory = await _full_memory_response_json(diarization=diarization)
    streaming = await _streaming_response_json(str(tmp_path), diarization=diarization)
    assert streaming == full_memory


@pytest.mark.parametrize("diarization", [False, True])
@pytest.mark.asyncio
async def test_sse_stream_is_byte_identical(tmp_path, diarization: bool):
    full_memory = await _full_memory_sse(diarization=diarization)
    streaming = await _streaming_sse(str(tmp_path), diarization=diarization)
    assert streaming == full_memory


@pytest.mark.asyncio
async def test_parity_fixture_actually_exercises_the_response(tmp_path):
    """Guard the parity assertions against trivially matching empty responses."""
    payload = json.loads(await _streaming_response_json(str(tmp_path), diarization=True))
    assert len(payload["segments"]) == len(_WINDOW_TEXTS)
    # Non-emptiness is the whole specification of this guard: it exists so the
    # byte-parity assertions above cannot pass on two equally empty responses.
    # An exact count would be asserting the fixture, not the guard.
    assert len(payload["word_segments"]) > 0, (  # falsegreen: ignore
        "words must be populated for parity to mean anything"
    )
    assert len(payload["raw_words"]) > 0, (  # falsegreen: ignore
        "raw words must be populated for parity to mean anything"
    )
    assert {item["speaker"] for item in payload["diarization"]} == {"2", "3"}
