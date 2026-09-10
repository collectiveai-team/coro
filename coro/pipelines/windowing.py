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

DEFAULT_WINDOW_SECONDS = 30.0
"""Window length every pipeline uses; part of the ASR window cache fingerprint."""

DEFAULT_OVERLAP_SECONDS = 2.0
"""Window overlap every pipeline uses; part of the ASR window cache fingerprint."""


# MARK: Result Model
@dataclass
class ASRWindowingResult:
    """Tokens accepted from ASR Windowing."""

    tokens: list[TranscriptToken]
    detected_language: str | None = None
    """Sticky auto-LID result for the whole call, if any -- see
    :class:`LanguageState`."""


# MARK: Sticky Auto-LID State
@dataclass
class LanguageState:
    """Request-scoped sticky auto-LID state for one ``transcribe_pcm``/``stream_pcm`` call.

    Lives with the request, not inside the shared (and now concurrent) ASR
    adapter instance: two overlapping requests must not see each other's
    detection. ``resolved`` is set by the first window whose detection
    succeeds and never changes after that -- see :func:`_resolve_window_language`.
    """

    resolved: str | None = None


async def _resolve_window_language(
    asr: Any, pcm: bytes, *, language: str | None, state: LanguageState | None
) -> str | None:
    """Return the language to decode one window with, running auto-LID if needed.

    - An explicit request ``language`` always wins outright; ``state`` is not
      even consulted, so a forced-language request never calls ``detect_language``.
    - No explicit language and no ``state`` (streaming/live pipelines, until
      they carry their own sticky state): unchanged behaviour -- ``None``
      reaches the adapter, whose own resolution/fallback applies.
    - No explicit language, ``state`` given, already resolved: the sticky
      language, no further detection calls.
    - No explicit language, ``state`` given, not yet resolved: run detection
      (only when ``asr`` exposes ``detect_language`` -- any backend without
      it, i.e. every backend but onnx-canary-split today, behaves exactly as
      before). A successful result becomes sticky and decodes *this* window;
      an unsuccessful one (not observed on real audio in the LID probe, but
      handled) decodes this window with ``None`` and detection is retried on
      the next window.
    """
    if language:
        return language
    if state is None:
        return None
    if state.resolved is not None:
        return state.resolved
    detect = getattr(asr, "detect_language", None)
    if detect is None:
        return None
    detected = await detect(pcm)
    if detected is not None:
        state.resolved = detected
    return detected


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

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    ) -> None:
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
        language_state: LanguageState | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Transcribe one window and emit its reconciled tokens.

        Every ASR Windowing path routes through here, including the tail flush,
        so token conversion, boundary reconciliation and prompt carry-over have
        exactly one implementation. ``language_state`` is ``None`` for any
        caller not yet carrying sticky auto-LID state (see
        :func:`_resolve_window_language`); today that is every caller except
        the Full-Memory Pipeline's ``transcribe_pcm``/``stream_pcm``.
        """
        logger.info(
            "asr_windowing window=%d start=%.2fs duration=%.2fs final=%s",
            plan.index,
            plan.offset_seconds,
            len(window) / BYTES_PER_SECOND,
            plan.is_final,
        )
        resolved_language = await _resolve_window_language(
            asr, window, language=language, state=language_state
        )
        asr_started = time.perf_counter()
        window_tokens = await asr.transcribe_pcm(
            window, language=resolved_language, prompt=carry.text
        )
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
        state = LanguageState()
        tokens: list[TranscriptToken] = []
        async for event in self.stream_pcm(
            pcm, asr=asr, language=language, prompt=prompt, language_state=state
        ):
            if isinstance(event, TokenBatchEvent):
                tokens.extend(event.tokens)
        return ASRWindowingResult(tokens=tokens, detected_language=state.resolved)

    # Streaming Transcription ----------------------------------------------
    async def stream_pcm(
        self,
        pcm: bytes,
        *,
        asr: Any,
        language: str | None = None,
        prompt: str | None = None,
        language_state: LanguageState | None = None,
    ) -> AsyncIterator[StreamEvent]:
        carry = _PromptCarry(text=prompt)
        for plan, window in self._plan_windows(pcm):
            async for event in self._run_window(
                window, plan, asr=asr, language=language, carry=carry, language_state=language_state
            ):
                yield event

    async def stream_chunks(
        self,
        chunks: AsyncIterator[bytes],
        *,
        asr: Any,
        language: str | None = None,
        prompt: str | None = None,
        language_state: LanguageState | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Window a chunk stream, matching the plan `_plan_windows` would produce.

        ``language_state``, when given, carries sticky auto-LID for the whole
        call exactly as ``stream_pcm``'s does -- see
        :func:`_resolve_window_language`. The Streaming and Live pipelines
        create one per request/connection and read ``.resolved`` back after
        (or during, for the Live pipeline) the stream to report it.

        A window is only dispatched once at least one byte beyond it has
        arrived, which is what proves it is not the final window. Emitting as
        soon as the buffer merely *reached* `window_bytes` could not know that,
        so a stream whose length was an exact multiple of the step dispatched
        the last full window as non-final and then ran a second window over the
        leftover overlap alone: a wasted inference whose region had already been
        covered, with the tail of the audio transcribed as an isolated
        overlap-length fragment instead of in the context of its full window.

        The cost is one chunk of lookahead. That is free while decoding a file
        (the next chunk is already available) and one client frame on the live
        socket, which pushes frames through unchanged.
        """
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

            while len(buffer) > self.window_bytes:
                window_count += 1
                plan = self._plan(window_count, consumed_bytes, is_final=False)
                window = bytes(buffer[: self.window_bytes])
                async for event in self._run_window(
                    window,
                    plan,
                    asr=asr,
                    language=language,
                    carry=carry,
                    language_state=language_state,
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
                bytes(buffer),
                plan,
                asr=asr,
                language=language,
                carry=carry,
                language_state=language_state,
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
