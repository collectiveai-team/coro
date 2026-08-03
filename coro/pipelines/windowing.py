"""Shared ASR Windowing for transcription pipelines."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
import logging
from math import inf
import time
from typing import Any

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import (
    StreamEvent,
    TokenBatchEvent,
    TranscriptDeltaEvent,
    TranscriptToken,
)

logger = logging.getLogger(__name__)

PROMPT_CARRY_TOKENS = 50
PROMPT_CARRY_CHARS = 200


# MARK: Result Model
@dataclass
class ASRWindowingResult:
    """Tokens accepted from ASR Windowing."""

    tokens: list[TranscriptToken]


# MARK: Overlap Token Acceptance
class OverlapTokenAcceptance:
    """Decide which tokens of an overlapping ASR window enter the transcript.

    Consecutive windows re-transcribe their shared overlap, so without a rule
    every word spoken there is emitted twice.  Each window is given a half-open
    acceptance region ``[boundary, cut)`` in absolute seconds; the regions of
    successive windows are contiguous, so every instant of audio belongs to
    exactly one window and each spoken word is emitted exactly once.

    The cut sits in the *middle* of the overlap rather than at either edge.  A
    word is therefore always claimed by a window that heard it whole: the window
    that owns it still has half the overlap of audio left after the word starts,
    and the next window already had half the overlap of audio before it.  A word
    straddling the cut is claimed by the earlier window, and the fragment the
    later window reports from its own leading edge falls before the boundary and
    is dropped.

    State is a single scalar, never a token history, so the Streaming Pipeline
    stays flat in memory however long the audio runs.
    """

    def __init__(self) -> None:
        self._accepted_through = 0.0

    def accept(
        self,
        tokens: Sequence[TranscriptToken],
        *,
        offset_seconds: float,
        cut_seconds: float,
    ) -> list[TranscriptToken]:
        """Offset a window's tokens and keep those inside its acceptance region."""
        boundary = self._accepted_through
        accepted = [
            TranscriptToken(
                start=token.start + offset_seconds,
                end=token.end + offset_seconds,
                text=token.text,
                probability=token.probability,
            )
            for token in tokens
            if boundary <= token.start + offset_seconds < cut_seconds
        ]
        self._accepted_through = max(boundary, cut_seconds)
        return accepted


@dataclass
class _WindowCarry:
    """O(1) state threaded through every window of one ASR Windowing run."""

    acceptance: OverlapTokenAcceptance
    prompt: str | None = None
    recent: deque[TranscriptToken] = field(
        default_factory=lambda: deque(maxlen=PROMPT_CARRY_TOKENS)
    )
    accepted_count: int = 0


# MARK: ASR Windowing
class ASRWindowing:
    """Transcribe PCM in overlapping windows behind a small interface."""

    def __init__(self, *, window_seconds: float = 30.0, overlap_seconds: float = 2.0) -> None:
        if overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be less than window_seconds")
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.window_bytes = self._seconds_to_bytes(window_seconds)
        self.overlap_bytes = self._seconds_to_bytes(overlap_seconds)
        self.step_bytes = self.window_bytes - self.overlap_bytes
        self.half_overlap_bytes = self._align(self.overlap_bytes // 2)
        self._stream_chunks_buffer_highwater = 0

    # Window Planning -------------------------------------------------------
    @staticmethod
    def _align(byte_count: int) -> int:
        return byte_count - (byte_count % BYTES_PER_SAMPLE)

    @classmethod
    def _seconds_to_bytes(cls, seconds: float) -> int:
        byte_count = int(SAMPLE_RATE * BYTES_PER_SAMPLE * seconds)
        return max(BYTES_PER_SAMPLE, cls._align(byte_count))

    @staticmethod
    def _seconds(byte_count: int) -> float:
        return byte_count / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    def _cut_seconds(self, offset_bytes: int, *, is_final: bool) -> float:
        """Return the absolute time at which this window stops being authoritative.

        Computed in bytes so the cut of one window and the offset of the next are
        derived from the same integer arithmetic and cannot drift apart.
        """
        if is_final:
            return inf
        return self._seconds(offset_bytes + self.step_bytes + self.half_overlap_bytes)

    def _windows(self, pcm: bytes) -> Iterator[tuple[int, bytes, bool]]:
        """Yield ``(offset_bytes, window, is_final)`` for a fully buffered PCM.

        A window is final when no audio follows it, which is the same condition
        the incremental path applies, so both paths plan identical windows.
        """
        if not pcm:
            return
        offset = 0
        while True:
            window = pcm[offset : offset + self.window_bytes]
            is_final = offset + self.window_bytes >= len(pcm)
            yield offset, window, is_final
            if is_final:
                return
            offset += self.step_bytes

    # Window Transcription --------------------------------------------------
    async def _emit_window(
        self,
        window: bytes,
        *,
        offset_bytes: int,
        is_final: bool,
        index: int,
        asr: Any,
        language: str | None,
        carry: _WindowCarry,
    ) -> AsyncIterator[StreamEvent]:
        """Transcribe one window and emit the events for its accepted tokens."""
        offset_seconds = self._seconds(offset_bytes)
        logger.info(
            "asr_windowing window=%d final=%s start=%.2fs duration=%.2fs",
            index,
            is_final,
            offset_seconds,
            self._seconds(len(window)),
        )
        asr_started = time.perf_counter()
        window_tokens = await asr.transcribe_pcm(window, language=language, prompt=carry.prompt)
        accepted = carry.acceptance.accept(
            window_tokens,
            offset_seconds=offset_seconds,
            cut_seconds=self._cut_seconds(offset_bytes, is_final=is_final),
        )
        logger.info(
            "asr_windowing window=%d asr_complete elapsed=%.3fs raw_tokens=%d accepted_tokens=%d",
            index,
            time.perf_counter() - asr_started,
            len(window_tokens),
            len(accepted),
        )
        if not accepted:
            return
        carry.recent.extend(accepted)
        carry.accepted_count += len(accepted)
        carry.prompt = "".join(token.text for token in carry.recent)[-PROMPT_CARRY_CHARS:]
        yield TokenBatchEvent(tokens=accepted)
        delta = "".join(token.text for token in accepted).strip()
        if delta:
            yield TranscriptDeltaEvent(delta=delta)

    # Batch Transcription ---------------------------------------------------
    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        asr: Any,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ASRWindowingResult:
        tokens: list[TranscriptToken] = []
        async for event in self.stream_pcm(pcm, asr=asr, language=language, prompt=prompt):
            if isinstance(event, TokenBatchEvent):
                tokens.extend(event.tokens)
        return ASRWindowingResult(tokens=tokens)

    # Streaming Transcription ----------------------------------------------
    async def stream_pcm(
        self,
        pcm: bytes,
        *,
        asr: Any,
        language: str | None = None,
        prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        carry = _WindowCarry(acceptance=OverlapTokenAcceptance(), prompt=prompt)
        for index, (offset_bytes, window, is_final) in enumerate(self._windows(pcm), start=1):
            async for event in self._emit_window(
                window,
                offset_bytes=offset_bytes,
                is_final=is_final,
                index=index,
                asr=asr,
                language=language,
                carry=carry,
            ):
                yield event

    async def stream_chunks(
        self,
        chunks: AsyncIterator[bytes],
        *,
        asr: Any,
        language: str | None = None,
        prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        carry = _WindowCarry(acceptance=OverlapTokenAcceptance(), prompt=prompt)
        buffer = bytearray()
        consumed_bytes = 0
        max_buffer = 0
        window_count = 0
        started = time.perf_counter()

        async for chunk in chunks:
            if not chunk:
                continue
            buffer.extend(chunk)
            max_buffer = max(max_buffer, len(buffer))

            # Strictly more than a window has arrived, so audio follows this one
            # and it cannot be the final window.  Equality is left to the tail
            # below, where the stream has ended and finality is known.
            while len(buffer) > self.window_bytes:
                window_count += 1
                async for event in self._emit_window(
                    bytes(buffer[: self.window_bytes]),
                    offset_bytes=consumed_bytes,
                    is_final=False,
                    index=window_count,
                    asr=asr,
                    language=language,
                    carry=carry,
                ):
                    yield event
                del buffer[: self.step_bytes]
                consumed_bytes += self.step_bytes

        if buffer:
            window_count += 1
            async for event in self._emit_window(
                bytes(buffer),
                offset_bytes=consumed_bytes,
                is_final=True,
                index=window_count,
                asr=asr,
                language=language,
                carry=carry,
            ):
                yield event

        self._stream_chunks_buffer_highwater = max_buffer
        logger.info(
            "asr_windowing complete elapsed=%.3fs windows=%d accepted_tokens=%d "
            "max_buffer_bytes=%d",
            time.perf_counter() - started,
            window_count,
            carry.accepted_count,
            max_buffer,
        )
