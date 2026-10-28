"""Streaming Pipeline sticky auto-LID: one detection result per request, reported everywhere."""

from __future__ import annotations

import struct
from unittest.mock import patch

import pytest

from coro.audio import AudioInput
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.streaming import StreamingPipeline
from coro.pipelines.windowing import ASRWindowing
from support.done_frame import render_done_frame


# ---------------------------------------------------------------------------
# Sticky auto-LID (ticket 05, streaming/live pipelines slice)
# ---------------------------------------------------------------------------


class _FakeAutoLIDAsr:
    """A canary-like fake exposing ``detect_language``, scripted per call."""

    def __init__(self, detections: list[str | None]) -> None:
        self._detections = list(detections)
        self.detect_calls = 0
        self.transcribe_languages: list[str | None] = []

    async def detect_language(self, pcm: bytes) -> str | None:
        detected = self._detections[self.detect_calls]
        self.detect_calls += 1
        return detected

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.transcribe_languages.append(language)
        return []


# 5 chunks of 0.5 s each = 2.5 s total: with 1.0 s windows and no overlap this
# plans exactly 3 windows (see test_full_memory_pipeline.py's identical note
# on ASRWindowing._seconds_to_bytes's byte-alignment floor).
_HALF_SECOND_CHUNK = struct.pack("<8000h", *([0] * 8000))


async def _three_window_stream(path: str, chunk_seconds: float = 1.0):
    for _ in range(5):
        yield _HALF_SECOND_CHUNK


def _mock_three_window_stream():
    return patch("coro.pipelines.streaming.stream_pcm_from_file", new=_three_window_stream)


@pytest.mark.asyncio
async def test_undeclared_language_sticks_after_the_first_successful_detection():
    """window 1 -> fallback, window 2 -> es (sticky), window 3 would say en but is forced es."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = StreamingPipeline(
        asr=asr, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    with _mock_three_window_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"))

    assert asr.detect_calls == 2  # window 3 never re-runs detection
    assert asr.transcribe_languages == [None, "es", "es"]
    assert result.detected_language == "es"


@pytest.mark.asyncio
async def test_explicit_language_skips_detection_entirely():
    asr = _FakeAutoLIDAsr(["es", "es"])  # would detect if ever consulted
    pipeline = StreamingPipeline(
        asr=asr, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    with _mock_three_window_stream():
        result = await pipeline.transcribe(AudioInput(b"audio"), language="fr")

    assert asr.detect_calls == 0
    assert asr.transcribe_languages == ["fr", "fr", "fr"]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_detected_language_is_reported_via_transcribe_source():
    """The lazy/spill Transcript Source also carries the sticky result."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = StreamingPipeline(
        asr=asr, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    with _mock_three_window_stream():
        source = await pipeline.transcribe_source(AudioInput(b"audio"))
    try:
        assert source.detected_language == "es"
    finally:
        source.close()


@pytest.mark.asyncio
async def test_detected_language_is_reported_on_the_streamed_done_frame():
    """verbose_json / equivalent reporting: the SSE done frame carries it too."""
    asr = _FakeAutoLIDAsr([None, "es"])
    pipeline = StreamingPipeline(
        asr=asr, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    done_frame = None
    with _mock_three_window_stream():
        async for event in pipeline.stream(AudioInput(b"audio")):
            if isinstance(event, StreamingDoneFrame):
                done_frame = event
    assert done_frame is not None
    body = render_done_frame(done_frame)
    assert body["detected_language"] == "es"
