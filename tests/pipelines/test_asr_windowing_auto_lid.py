"""ASR Windowing sticky auto-LID: detection runs until it succeeds, then the language sticks."""

from __future__ import annotations

import pytest

from coro.pipelines.windowing import ASRWindowing, LanguageState
from support.windowing import (
    FakeASR,
    ScriptedLIDASR,
    async_chunks,
    grid_chunks,
    pcm_seconds,
)


def one_second_windows() -> ASRWindowing:
    """Return an ASR Windowing that cuts 2.5 s of PCM into three windows."""
    return ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)


@pytest.mark.asyncio
async def test_transcribe_pcm_sticks_to_the_first_successful_detection():
    asr = ScriptedLIDASR([None, "es"])

    result = await one_second_windows().transcribe_pcm(
        pcm_seconds(2.5), asr=asr, language=None, prompt=None
    )

    assert asr.languages == [None, "es", "es"]
    assert result.detected_language == "es"


@pytest.mark.asyncio
async def test_no_successful_detection_falls_back_every_window_and_reports_none():
    asr = ScriptedLIDASR([None, None, None])

    result = await one_second_windows().transcribe_pcm(
        pcm_seconds(2.5), asr=asr, language=None, prompt=None
    )

    assert asr.languages == [None, None, None]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_an_unsuccessful_detection_is_retried_on_the_next_window():
    asr = ScriptedLIDASR([None, None, "es"])

    result = await one_second_windows().transcribe_pcm(
        pcm_seconds(2.5), asr=asr, language=None, prompt=None
    )

    assert asr.languages == [None, None, "es"]
    assert result.detected_language == "es"


@pytest.mark.asyncio
async def test_explicit_language_never_calls_detect_language():
    asr = ScriptedLIDASR([])

    result = await one_second_windows().transcribe_pcm(
        pcm_seconds(2.5), asr=asr, language="fr", prompt=None
    )

    assert asr.languages == ["fr", "fr", "fr"]
    assert result.detected_language is None


@pytest.mark.asyncio
async def test_a_backend_without_detect_language_reports_no_detected_language():
    result = await one_second_windows().transcribe_pcm(
        pcm_seconds(1.0), asr=FakeASR(), language=None, prompt=None
    )

    assert result.detected_language is None


@pytest.mark.asyncio
async def test_stream_chunks_without_a_language_state_never_detects():
    asr = ScriptedLIDASR([])

    async for _ in one_second_windows().stream_chunks(
        async_chunks([pcm_seconds(2.5)]), asr=asr, language=None, prompt=None
    ):
        pass

    assert asr.languages == [None, None, None]


@pytest.mark.asyncio
async def test_stream_chunks_sticks_to_the_first_successful_detection():
    asr = ScriptedLIDASR([None, "es"])
    state = LanguageState()

    async for _ in one_second_windows().stream_chunks(
        async_chunks([pcm_seconds(2.5)]), asr=asr, language=None, prompt=None, language_state=state
    ):
        pass

    assert asr.languages == [None, "es", "es"]
    assert state.resolved == "es"


@pytest.mark.asyncio
async def test_stream_chunks_prompt_carry_over_matches_stream_pcm():
    by_chunks = FakeASR()
    by_pcm = FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)
    pcm = pcm_seconds(2.0)

    async for _ in windowing.stream_pcm(pcm, asr=by_pcm, language="es", prompt="initial"):
        pass
    async for _ in windowing.stream_chunks(
        async_chunks(grid_chunks(pcm)), asr=by_chunks, language="es", prompt="initial"
    ):
        pass

    assert by_chunks.prompts == by_pcm.prompts
