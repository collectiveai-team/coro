"""Shared ASR fakes and PCM helpers for the ASR Windowing tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import TranscriptToken

BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE


class FakeASR:
    """Emit one token at the midpoint of whichever window it is handed.

    Mid-window placement keeps the token clear of every reconciliation
    boundary, so tests about prompt carry-over or event shape stay independent
    of overlap reconciliation.
    """

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
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
