"""ASR Windowing deep module behavior."""

from __future__ import annotations

import pytest

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asr_diar_server.core.types import TranscriptToken
from asr_diar_server.pipelines.windowing import ASRWindowing


class _FakeASR:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        self.prompts.append(prompt)
        call = len(self.prompts)
        return [TranscriptToken(start=0.0, end=0.25, text=f" word{call}", probability=1.0)]


def _pcm_seconds(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


@pytest.mark.asyncio
async def test_asr_windowing_calls_adapter_for_overlapping_windows():
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    result = await windowing.transcribe_pcm(
        _pcm_seconds(2.0),
        asr=asr,
        language="es",
        prompt="hint",
    )

    assert len(asr.prompts) == 3
    assert [token.text for token in result.tokens] == [" word1", " word2", " word3"]


@pytest.mark.asyncio
async def test_asr_windowing_streams_delta_per_accepted_window():
    asr = _FakeASR()
    windowing = ASRWindowing(window_seconds=1.0, overlap_seconds=0.25)

    events = [
        event
        async for event in windowing.stream_pcm(
            _pcm_seconds(1.2),
            asr=asr,
            language=None,
            prompt=None,
        )
    ]

    delta_events = [event for event in events if event["type"] == "transcript.text.delta"]
    assert [event["delta"] for event in delta_events] == ["word1", "word2"]


def test_window_bytes_are_even_for_pcm_alignment():
    windowing = ASRWindowing(window_seconds=0.01, overlap_seconds=0.0)

    assert windowing.window_bytes % BYTES_PER_SAMPLE == 0
