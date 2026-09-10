"""Full-Memory Pipeline orchestration with fake ASR and diarization adapters.

Tests verify that the FullMemoryPipeline:
- Calls the ASR adapter with PCM bytes converted from the audio input.
- Calls the optional diarization adapter.
- Passes resulting tokens and timeline to the core response builder.
- Returns a valid transcription response dict.
- Propagates prompt to ASR adapter.
- Handles empty token output without crashing.

The audio conversion step (ffmpeg) is mocked so tests run without
subprocess or a real audio container.
"""

from __future__ import annotations

import struct
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from coro.audio import AudioInput
from coro.core.models import SpeakerSegment, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from coro.pipelines.windowing import ASRWindowing

RESPONSE_KEYS = {
    "segments",
    "word_segments",
    "transcript",
    "diarization",
    "raw_words",
    "detected_language",
}

_FAKE_PCM = struct.pack("<1600h", *([0] * 1600))


def _mock_convert(return_value: bytes):
    """Return a context manager that patches the path-based PCM conversion."""
    return patch(
        "coro.pipelines.full_memory.convert_path_to_pcm_bytes",
        new=AsyncMock(return_value=return_value),
    )


class _FakeASRAdapter:
    """Return a fixed token list from transcribe_pcm."""

    def __init__(self, tokens=None):
        self._tokens = tokens or []
        self.last_prompt = None
        self.last_language = None

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        self.last_prompt = prompt
        self.last_language = language
        return list(self._tokens)


class _FakeDiarizationAdapter:
    def __init__(self, timeline=None):
        self._timeline = timeline or []

    async def diarize_pcm(self, pcm: bytes) -> list[SpeakerSegment]:
        return list(self._timeline)


@pytest.mark.asyncio
async def test_full_memory_pipeline_returns_response_shape():
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert set(asdict(result)) == RESPONSE_KEYS


@pytest.mark.asyncio
async def test_full_memory_pipeline_passes_prompt_to_asr():
    asr = _FakeASRAdapter()
    pipeline = FullMemoryPipeline(asr=asr, diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.transcribe(AudioInput(b"audio"), prompt="test prompt")
    assert asr.last_prompt == "test prompt"


@pytest.mark.asyncio
async def test_full_memory_pipeline_passes_language_to_asr():
    asr = _FakeASRAdapter()
    pipeline = FullMemoryPipeline(asr=asr, diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.transcribe(AudioInput(b"audio"), language="es")
    assert asr.last_language == "es"


@pytest.mark.asyncio
async def test_full_memory_pipeline_uses_diarization_when_provided():
    tokens = [TranscriptToken(start=0.0, end=1.0, text=" hola.", probability=0.9)]
    timeline = [SpeakerSegment(start=0.0, end=2.0, speaker=1)]
    asr = _FakeASRAdapter(tokens=tokens)
    diar = _FakeDiarizationAdapter(timeline=timeline)
    pipeline = FullMemoryPipeline(asr=asr, diarization=diar)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))
    seg = result.segments[0]
    assert seg.speaker == "1"


@pytest.mark.asyncio
async def test_full_memory_pipeline_empty_tokens_no_crash():
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(tokens=[]), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))
    assert result.segments == []
    assert result.raw_words == []


@pytest.mark.asyncio
async def test_full_memory_pipeline_cleans_up_the_upload_temp_file():
    """Uploads are spooled eagerly, so the consuming pipeline must unlink them.

    Without this the file would outlive every request and a long upload would
    leak its full size to disk each time.
    """
    audio = AudioInput(b"audio")
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_convert(_FAKE_PCM):
        await pipeline.transcribe(audio)

    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_full_memory_pipeline_cleans_up_when_transcription_fails():
    """A mid-pipeline failure still releases the upload temp file."""

    class _ExplodingASR:
        async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
            raise RuntimeError("asr exploded")

    audio = AudioInput(b"audio")
    pipeline = FullMemoryPipeline(asr=_ExplodingASR(), diarization=None)
    with _mock_convert(_FAKE_PCM), pytest.raises(RuntimeError, match="asr exploded"):
        await pipeline.transcribe(audio)

    assert audio._temp_path is None


@pytest.mark.asyncio
async def test_full_memory_pipeline_stream_cleans_up_the_upload_temp_file():
    """The streaming path owns cleanup too, once the generator is exhausted."""
    audio = AudioInput(b"audio")
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_convert(_FAKE_PCM):
        async for _ in pipeline.stream(audio):
            pass

    assert audio._temp_path is None


# ---------------------------------------------------------------------------
# Sticky auto-LID (ticket 05, core + Full-Memory Pipeline slice)
# ---------------------------------------------------------------------------


class _FakeAutoLIDAsr:
    """A canary-like fake exposing ``detect_language``, scripted per call.

    Mirrors what ``OnnxCanarySplitASRAdapter.detect_language`` promises: a
    per-window detection result (or ``None``), independent of
    ``transcribe_pcm``'s own language handling.
    """

    def __init__(
        self, detections: list[str | None], *, tokens: list[TranscriptToken] | None = None
    ):
        self._detections = list(detections)
        self._tokens = tokens or []
        self.detect_calls = 0
        self.transcribe_languages: list[str | None] = []

    async def detect_language(self, pcm: bytes) -> str | None:
        detected = self._detections[self.detect_calls]
        self.detect_calls += 1
        return detected

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.transcribe_languages.append(language)
        return list(self._tokens)


def _three_window_pcm() -> bytes:
    """2.5 s of PCM: with 1.0 s windows and no overlap, this plans exactly 3
    windows (a whole-multiple duration would add a razor-thin trailing window
    -- the windowing byte-alignment floor never lets overlap_bytes reach
    exactly zero, see ``ASRWindowing._seconds_to_bytes``)."""
    return struct.pack("<40000h", *([0] * 40000))


@pytest.mark.asyncio
async def test_undeclared_language_sticks_after_the_first_successful_detection():
    """window 1 -> fallback, window 2 -> es (sticky), window 3 would say en but is forced es."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = FullMemoryPipeline(
        asr=asr, diarization=None, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    with _mock_convert(_three_window_pcm()):
        result = await pipeline.transcribe(AudioInput(b"audio"))

    assert asr.detect_calls == 2  # window 3 never re-runs detection
    assert asr.transcribe_languages == [None, "es", "es"]
    assert result.detected_language == "es"


@pytest.mark.asyncio
async def test_explicit_language_skips_detection_entirely():
    """A forced request language never calls detect_language, on any window."""
    asr = _FakeAutoLIDAsr(["es", "es"])  # would detect if ever consulted
    pipeline = FullMemoryPipeline(
        asr=asr, diarization=None, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    with _mock_convert(_three_window_pcm()):
        result = await pipeline.transcribe(AudioInput(b"audio"), language="fr")

    assert asr.detect_calls == 0
    assert asr.transcribe_languages == ["fr", "fr", "fr"]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_a_backend_without_detect_language_reports_no_detected_language():
    """Every backend but onnx-canary-split (no auto-LID) is unaffected."""
    pipeline = FullMemoryPipeline(asr=_FakeASRAdapter(), diarization=None)
    with _mock_convert(_FAKE_PCM):
        result = await pipeline.transcribe(AudioInput(b"audio"))

    assert result.detected_language is None
