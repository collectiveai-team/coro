"""LiveTranscriptionSession: sticky auto-LID per connection (ticket 05).

Drives the session directly rather than through the websocket route, so a
multi-window scenario runs in milliseconds without a 90+ second audio stream.
The websocket-level tests (``tests/api/deepgram/test_listen_websocket.py``)
cover the negotiate-time explicit-language-skips-detection behaviour and the
closing ``Metadata`` frame's reporting.
"""

from __future__ import annotations

import pytest

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.pipelines.live import LiveAudioSource, LiveTranscriptionSession
from coro.pipelines.windowing import ASRWindowing

_BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


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


def _seconds(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


async def _feed(source: LiveAudioSource, pcm: bytes, *, chunk_seconds: float = 0.5) -> None:
    chunk_bytes = int(_BYTES_PER_SECOND * chunk_seconds)
    for offset in range(0, len(pcm), chunk_bytes):
        await source.push(pcm[offset : offset + chunk_bytes])
    await source.close()


@pytest.mark.asyncio
async def test_sticky_language_persists_across_windows_in_one_connection():
    """window 1 -> fallback, window 2 -> es (sticky), window 3 would say en but is forced es."""
    asr = _FakeAutoLIDAsr([None, "es"])
    session = LiveTranscriptionSession(
        asr=asr, windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    source = LiveAudioSource()

    await _feed(source, _seconds(2.5))
    async for _ in session.run(source):
        pass

    assert asr.detect_calls == 2  # window 3 never re-runs detection
    assert asr.transcribe_languages == [None, "es", "es"]
    assert session.detected_language == "es"


@pytest.mark.asyncio
async def test_explicit_language_never_calls_detect_language():
    asr = _FakeAutoLIDAsr(["es", "es"])  # would detect if ever consulted
    session = LiveTranscriptionSession(
        asr=asr,
        windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0),
        language="fr",
    )
    source = LiveAudioSource()

    await _feed(source, _seconds(2.5))
    async for _ in session.run(source):
        pass

    assert asr.detect_calls == 0
    assert asr.transcribe_languages == ["fr", "fr", "fr"]
    assert session.detected_language is None


@pytest.mark.asyncio
async def test_a_backend_without_detect_language_is_unaffected():
    class _FakeASR:
        async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
            return []

    session = LiveTranscriptionSession(
        asr=_FakeASR(), windowing=ASRWindowing(window_seconds=1.0, overlap_seconds=0.0)
    )
    source = LiveAudioSource()

    await _feed(source, _seconds(1.0))
    async for _ in session.run(source):
        pass

    assert session.detected_language is None
