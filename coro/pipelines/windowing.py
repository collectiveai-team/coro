"""Shared ASR Windowing for transcription pipelines."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
import logging
import math
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

BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
PROMPT_TOKEN_LIMIT = 50
PROMPT_CHAR_LIMIT = 200


# MARK: Result Model
@dataclass
class ASRWindowingResult:
    """Tokens accepted from ASR Windowing."""

    tokens: list[TranscriptToken]


# MARK: Window Plan
@dataclass(frozen=True)
class _WindowPlan:
    """Placement of one window on the absolute audio timeline.

    ``accept_from`` and ``accept_until`` bound the half-open span of absolute
    audio time this window is authoritative for. A window's span begins at the
    midpoint of the region it shares with the previous window and ends at the
    midpoint of the region it shares with the next one, so adjacent spans meet
    exactly: every region of audio is attributed to exactly one window, with no
    duplicated and no dropped region.
    """

    index: int
    offset_seconds: float
    accept_from: float
    accept_until: float
    is_final: bool


# MARK: Prompt Carry
@dataclass
class _PromptCarry:
    """Bounded prompt carry-over built only from reconciled tokens.

    Retaining the whole transcript would grow memory O(audio length); the
    bounded deque keeps the carry-over identical (the last
    ``PROMPT_TOKEN_LIMIT`` accepted tokens) at constant memory.
    """

    text: str | None = None
    _tokens: deque[TranscriptToken] = field(
        init=False,
        default_factory=lambda: deque(maxlen=PROMPT_TOKEN_LIMIT),
    )

    def extend(self, tokens: list[TranscriptToken]) -> None:
        """Fold newly accepted tokens into the bounded carry-over text."""
        self._tokens.extend(tokens)
        self.text = "".join(token.text for token in self._tokens)[-PROMPT_CHAR_LIMIT:]


def _reconcile(window_tokens: list[Any], plan: _WindowPlan) -> list[TranscriptToken]:
    """Shift window-relative tokens onto the absolute timeline, once each.

    A token is kept only when its absolute start falls inside the window's
    authoritative span, so a region two adjacent windows share contributes
    tokens exactly once instead of twice.
    """
    accepted: list[TranscriptToken] = []
    for token in window_tokens:
        start = token.start + plan.offset_seconds
        if start < plan.accept_from or start >= plan.accept_until:
            continue
        accepted.append(
            TranscriptToken(
                start=start,
                end=token.end + plan.offset_seconds,
                text=token.text,
                probability=token.probability,
            )
        )
    return accepted


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
        # Derived from the byte-aligned sizes, not the requested seconds, so
        # reconciliation boundaries land exactly where windows actually sit.
        self._step_seconds = self.step_bytes / BYTES_PER_SECOND
        self._half_overlap_seconds = self.overlap_bytes / BYTES_PER_SECOND / 2.0

    # Window Planning -------------------------------------------------------
    @staticmethod
    def _seconds_to_bytes(seconds: float) -> int:
        byte_count = int(SAMPLE_RATE * BYTES_PER_SAMPLE * seconds)
        return max(BYTES_PER_SAMPLE, byte_count - (byte_count % BYTES_PER_SAMPLE))

    def _plan(self, index: int, offset_bytes: int, *, is_final: bool) -> _WindowPlan:
        """Describe where a window sits and which region it is authoritative for."""
        offset_seconds = offset_bytes / BYTES_PER_SECOND
        boundary_before = offset_seconds + self._half_overlap_seconds
        boundary_after = offset_seconds + self._step_seconds + self._half_overlap_seconds
        return _WindowPlan(
            index=index,
            offset_seconds=offset_seconds,
            accept_from=-math.inf if index == 1 else boundary_before,
            accept_until=math.inf if is_final else boundary_after,
            is_final=is_final,
        )

    def _plan_windows(self, pcm: bytes) -> Iterator[tuple[_WindowPlan, bytes]]:
        if not pcm:
            return
        offset = 0
        index = 0
        while offset < len(pcm):
            window = pcm[offset : offset + self.window_bytes]
            is_final = offset + self.window_bytes >= len(pcm)
            index += 1
            yield self._plan(index, offset, is_final=is_final), window
            if is_final:
                break
            offset += self.step_bytes

    # Window Execution ------------------------------------------------------
    async def _run_window(
        self,
        window: bytes,
        plan: _WindowPlan,
        *,
        asr: Any,
        language: str | None,
        carry: _PromptCarry,
    ) -> AsyncIterator[StreamEvent]:
        """Transcribe one window and emit its reconciled tokens.

        Every ASR Windowing path routes through here, including the tail flush,
        so token conversion, boundary reconciliation and prompt carry-over have
        exactly one implementation.
        """
        logger.info(
            "asr_windowing window=%d start=%.2fs duration=%.2fs final=%s",
            plan.index,
            plan.offset_seconds,
            len(window) / BYTES_PER_SECOND,
            plan.is_final,
        )
        asr_started = time.perf_counter()
        window_tokens = await asr.transcribe_pcm(window, language=language, prompt=carry.text)
        logger.info(
            "asr_windowing window=%d asr_complete elapsed=%.3fs raw_tokens=%d",
            plan.index,
            time.perf_counter() - asr_started,
            len(window_tokens),
        )
        accepted = _reconcile(window_tokens, plan)
        if not accepted:
            return
        carry.extend(accepted)
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
        carry = _PromptCarry(text=prompt)
        for plan, window in self._plan_windows(pcm):
            async for event in self._run_window(
                window, plan, asr=asr, language=language, carry=carry
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
        buffer = bytearray()
        consumed_bytes = 0
        carry = _PromptCarry(text=prompt)
        accepted_count = 0
        max_buffer = 0
        window_count = 0
        started = time.perf_counter()

        async for chunk in chunks:
            if not chunk:
                continue
            buffer.extend(chunk)
            if len(buffer) > max_buffer:
                max_buffer = len(buffer)

            while len(buffer) >= self.window_bytes:
                window_count += 1
                plan = self._plan(window_count, consumed_bytes, is_final=False)
                window = bytes(buffer[: self.window_bytes])
                async for event in self._run_window(
                    window, plan, asr=asr, language=language, carry=carry
                ):
                    if isinstance(event, TokenBatchEvent):
                        accepted_count += len(event.tokens)
                    yield event

                del buffer[: self.step_bytes]
                consumed_bytes += self.step_bytes

        if buffer:
            window_count += 1
            plan = self._plan(window_count, consumed_bytes, is_final=True)
            async for event in self._run_window(
                bytes(buffer), plan, asr=asr, language=language, carry=carry
            ):
                if isinstance(event, TokenBatchEvent):
                    accepted_count += len(event.tokens)
                yield event

        self._stream_chunks_buffer_highwater = max_buffer
        logger.info(
            "asr_windowing complete elapsed=%.3fs windows=%d accepted_tokens=%d "
            "max_buffer_bytes=%d",
            time.perf_counter() - started,
            window_count,
            accepted_count,
            max_buffer,
        )
