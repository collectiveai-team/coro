"""Streaming Pipeline sticky auto-LID: one detection result per request, reported everywhere."""

from __future__ import annotations

import pytest

from coro.audio import AudioInput
from coro.core.models import TranscriptToken
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing
from support.done_frame import render_done_frame
from support.factories import make_wav


# ---------------------------------------------------------------------------
# Sticky auto-LID (ticket 05, streaming/live pipelines slice)
# ---------------------------------------------------------------------------


class _FakeAutoLIDAsr:
    """An in-memory canary-like ASR Adapter exposing ``detect_language``, scripted per call."""

    honours_prompt = True

    def __init__(self, detections: list[str | None]) -> None:
        self._detections = list(detections)
        self.detect_calls = 0
        self.transcribe_languages: list[str | None] = []

    async def detect_language(self, pcm: bytes) -> str | None:
        detected = self._detections[self.detect_calls]
        self.detect_calls += 1
        return detected

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        self.transcribe_languages.append(language)
        return []


def _three_window_audio() -> AudioInput:
    """A real 2.5 s WAV: with 1.0 s windows and no overlap ffmpeg's PCM plans 3 windows."""
    return AudioInput(make_wav(frames=40000), filename="clip.wav")


def _pipeline(asr: _FakeAutoLIDAsr) -> StreamingPipeline:
    return StreamingPipeline(
        asr=asr, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )


@pytest.mark.asyncio
async def test_undeclared_language_sticks_after_the_first_successful_detection():
    """window 1 -> fallback, window 2 -> es (sticky), window 3 would say en but is forced es."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = _pipeline(asr)
    result = await pipeline.transcribe(_three_window_audio())

    assert asr.detect_calls == 2  # window 3 never re-runs detection
    assert asr.transcribe_languages == [None, "es", "es"]
    assert result.detected_language == "es"


@pytest.mark.asyncio
async def test_explicit_language_skips_detection_entirely():
    asr = _FakeAutoLIDAsr(["es", "es"])  # would detect if ever consulted
    pipeline = _pipeline(asr)
    result = await pipeline.transcribe(_three_window_audio(), language="fr")

    assert asr.detect_calls == 0
    assert asr.transcribe_languages == ["fr", "fr", "fr"]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_detected_language_is_reported_via_transcribe_source():
    """The lazy/spill Transcript Source also carries the sticky result."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = _pipeline(asr)
    source = await pipeline.transcribe_source(_three_window_audio())
    try:
        assert source.detected_language == "es"
    finally:
        source.close()


@pytest.mark.asyncio
async def test_detected_language_is_reported_on_the_streamed_done_frame():
    """verbose_json / equivalent reporting: the SSE done frame carries it too."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = _pipeline(asr)
    done_frame = None
    async for event in pipeline.stream(_three_window_audio()):
        if isinstance(event, StreamingDoneFrame):
            done_frame = event
    assert done_frame is not None
    body = render_done_frame(done_frame)
    assert body["detected_language"] == "es"
