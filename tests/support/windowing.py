"""In-memory ASR Adapters and PCM helpers for the ASR Windowing tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import TranscriptToken

BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


class FakeASR:
    """An ASR Adapter emitting one token at the midpoint of each window it is handed.

    Mid-window placement keeps the token clear of every reconciliation
    boundary, so tests about prompt carry-over or event shape stay independent
    of overlap reconciliation.
    """

    honours_prompt = True

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Return one token named after the call number, at the window midpoint."""
        self.prompts.append(prompt)
        call = len(self.prompts)
        midpoint = len(pcm) / BYTES_PER_SECOND / 2
        return [
            TranscriptToken(
                start=midpoint,
                end=midpoint + 0.25,
                text=f" word{call}",
                probability=1.0,
            )
        ]


class WindowRecordingASR:
    """An ASR Adapter that keeps the exact window bytes it is handed and hears nothing."""

    honours_prompt = True

    def __init__(self) -> None:
        self.windows: list[bytes] = []

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Keep the window and return no tokens."""
        self.windows.append(pcm)
        return []


class CountingChunkSource:
    """A chunk stream that counts how many bytes its consumer has pulled so far."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.pulled_bytes = 0

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield the chunks in order, counting each one as it is handed out."""
        for chunk in self._chunks:
            self.pulled_bytes += len(chunk)
            yield chunk


def pcm_seconds(seconds: float) -> bytes:
    """Return ``seconds`` of silent canonical PCM."""
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


async def async_chunks(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Yield ``chunks`` as an async stream."""
    for chunk in chunks:
        yield chunk


def grid_chunks(pcm: bytes) -> list[bytes]:
    """Cut ``pcm`` into 0.4 s chunks."""
    chunk_size = int(BYTES_PER_SECOND * 0.4)
    return [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]
