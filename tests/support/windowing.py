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


class OverlapEchoASR:
    """An ASR Adapter emitting one token at the first instant of every window.

    For every window after the first, that instant lies in the region shared
    with the preceding window, which already transcribed it.
    """

    honours_prompt = True

    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Return one token named after the call number, at the window start."""
        self.prompts.append(prompt)
        return [
            TranscriptToken(start=0.0, end=0.25, text=f" word{len(self.prompts)}", probability=1.0)
        ]


def word_grid(seconds: float, *, origin: float = 0.05, step: float = 0.1) -> list[float]:
    """Return the absolute start time of each word ``w<index>`` of a grid of ``seconds``."""
    count = int((seconds - origin) / step) + 1
    return [origin + step * index for index in range(count) if origin + step * index < seconds]


class TimelineGridASR:
    """An ASR Adapter that hears a fixed grid of words placed on the absolute timeline.

    Window ``n`` starts at ``n * step_seconds`` on every ASR Windowing path, so
    the fake maps its window back onto the timeline and returns every word that
    falls inside it, including the ones an overlapping neighbour also hears.
    """

    honours_prompt = True

    def __init__(self, step_seconds: float, grid: list[float]) -> None:
        self.step_seconds = step_seconds
        self.grid = grid
        self.windows = 0
        self.heard_tokens = 0

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Return the grid words inside this window, in window-local time."""
        offset = self.windows * self.step_seconds
        self.windows += 1
        duration = len(pcm) / BYTES_PER_SECOND
        tokens = [
            TranscriptToken(
                start=at - offset, end=at - offset + 0.05, text=f" w{index}", probability=1.0
            )
            for index, at in enumerate(self.grid)
            if offset <= at < offset + duration
        ]
        self.heard_tokens += len(tokens)
        return tokens


class ScriptedLIDASR:
    """An ASR Adapter with auto-LID whose detections follow a script, one per call.

    A detection asked for past the end of the script fails the test, so the
    script length is the number of detections a test allows.
    """

    honours_prompt = False

    def __init__(self, detections: list[str | None]) -> None:
        self.detections = list(detections)
        self.languages: list[str | None] = []

    async def detect_language(self, pcm: bytes) -> str | None:
        """Return the next scripted detection."""
        if not self.detections:
            raise AssertionError("detect_language was asked past the end of its script")
        return self.detections.pop(0)

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        """Keep the language the window was transcribed in and return no tokens."""
        self.languages.append(language)
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
