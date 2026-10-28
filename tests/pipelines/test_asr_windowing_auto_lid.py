"""ASR Windowing sticky auto-LID: detection runs until it succeeds, then the language sticks."""

from __future__ import annotations

import pytest

from coro.pipelines.windowing import ASRWindowing, LanguageState
from support.windowing import (
    FakeASR as _FakeASR,
    async_chunks as _async_chunks,
    grid_chunks as _grid_chunks,
    pcm_seconds as _pcm_seconds,
)


# ---------------------------------------------------------------------------
# Sticky auto-LID (ticket 05, core + Full-Memory Pipeline slice)
# ---------------------------------------------------------------------------


class _AutoLIDASR:
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


@pytest.mark.asyncio
async def test_transcribe_pcm_sticks_to_the_first_successful_detection():
    asr = _AutoLIDASR([None, "es"])
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)

    result = await windowing.transcribe_pcm(_pcm_seconds(2.5), asr=asr, language=None, prompt=None)

    assert asr.detect_calls == 2  # window 3 never re-runs detection
    assert asr.transcribe_languages == [None, "es", "es"]
    assert result.detected_language == "es"


@pytest.mark.asyncio
async def test_no_successful_detection_falls_back_every_window_and_reports_none():
    asr = _AutoLIDASR([None, None, None])
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)

    result = await windowing.transcribe_pcm(_pcm_seconds(2.5), asr=asr, language=None, prompt=None)

    assert asr.detect_calls == 3
    assert asr.transcribe_languages == [None, None, None]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_explicit_language_never_calls_detect_language():
    asr = _AutoLIDASR(["es", "es", "es"])  # would detect if ever consulted
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)

    result = await windowing.transcribe_pcm(_pcm_seconds(2.5), asr=asr, language="fr", prompt=None)

    assert asr.detect_calls == 0
    assert asr.transcribe_languages == ["fr", "fr", "fr"]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_a_backend_without_detect_language_is_unaffected():
    """Every backend but onnx-canary-split: unchanged, None reaches transcribe_pcm."""
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)

    result = await windowing.transcribe_pcm(_pcm_seconds(1.0), asr=asr, language=None, prompt=None)

    assert result.detected_language is None


@pytest.mark.asyncio
async def test_stream_chunks_without_a_language_state_behaves_as_before_auto_lid():
    """No language_state passed (a caller not yet wired for auto-LID): unchanged."""
    asr = _AutoLIDASR(["es", "es", "es"])
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    pcm = _pcm_seconds(2.5)

    async for _ in windowing.stream_chunks(
        _async_chunks([pcm]), asr=asr, language=None, prompt=None
    ):
        pass

    assert asr.detect_calls == 0
    assert asr.transcribe_languages == [None, None, None]


@pytest.mark.asyncio
async def test_stream_chunks_sticks_to_the_first_successful_detection():
    """stream_chunks honours a passed language_state exactly like stream_pcm does."""
    asr = _AutoLIDASR([None, "es"])
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    pcm = _pcm_seconds(2.5)
    state = LanguageState()

    async for _ in windowing.stream_chunks(
        _async_chunks([pcm]), asr=asr, language=None, prompt=None, language_state=state
    ):
        pass

    assert asr.detect_calls == 2  # window 3 never re-runs detection
    assert asr.transcribe_languages == [None, "es", "es"]
    assert state.resolved == "es"


@pytest.mark.asyncio
async def test_stream_chunks_prompt_carry_over_matches_stream_pcm():
    asr_chunks = _FakeASR()
    asr_pcm = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = _pcm_seconds(2.0)
    chunks = _grid_chunks(pcm)

    [e async for e in windowing.stream_pcm(pcm, asr=asr_pcm, language="es", prompt="initial")]
    [
        e
        async for e in windowing.stream_chunks(
            _async_chunks(chunks), asr=asr_chunks, language="es", prompt="initial"
        )
    ]

    assert asr_chunks.prompts == asr_pcm.prompts
