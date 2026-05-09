"""Shared ASR Windowing for transcription pipelines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from asr_diar_server.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asr_diar_server.core.types import (
    StreamEvent,
    TokenBatchEvent,
    TranscriptDeltaEvent,
    TranscriptToken,
)


# MARK: Result Model
@dataclass
class ASRWindowingResult:
    """Tokens accepted from ASR Windowing."""

    tokens: list[TranscriptToken]


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

    # Window Planning -------------------------------------------------------
    @staticmethod
    def _seconds_to_bytes(seconds: float) -> int:
        byte_count = int(SAMPLE_RATE * BYTES_PER_SAMPLE * seconds)
        return max(BYTES_PER_SAMPLE, byte_count - (byte_count % BYTES_PER_SAMPLE))

    def _windows(self, pcm: bytes):
        if not pcm:
            return
        offset = 0
        while offset < len(pcm):
            window = pcm[offset : offset + self.window_bytes]
            if window:
                yield offset / (SAMPLE_RATE * BYTES_PER_SAMPLE), window
            if offset + self.window_bytes >= len(pcm):
                break
            offset += self.step_bytes

    # Batch Transcription via chunk iterator ------------------------------------
    async def transcribe_chunks(
        self,
        chunks,
        *,
        asr: Any,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ASRWindowingResult:
        """Consume an async chunk iterator and return all tokens."""
        tokens: list[TranscriptToken] = []
        async for event in self.stream_chunks(chunks, asr=asr, language=language, prompt=prompt):
            if isinstance(event, TokenBatchEvent):
                tokens.extend(event.tokens)
        return ASRWindowingResult(tokens=tokens)

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
        prompt_carry = prompt
        async for event in self.stream_pcm(pcm, asr=asr, language=language, prompt=prompt):
            if isinstance(event, TokenBatchEvent):
                tokens.extend(event.tokens)
                prompt_carry = "".join(token.text for token in tokens[-50:])[-200:] or prompt_carry
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
        tokens: list[TranscriptToken] = []
        prompt_carry = prompt
        for offset_seconds, window in self._windows(pcm):
            window_tokens = await asr.transcribe_pcm(
                window,
                language=language,
                prompt=prompt_carry,
            )
            accepted = [
                TranscriptToken(
                    start=token.start + offset_seconds,
                    end=token.end + offset_seconds,
                    text=token.text,
                    probability=token.probability,
                )
                for token in window_tokens
            ]
            if not accepted:
                continue
            tokens.extend(accepted)
            prompt_carry = "".join(token.text for token in tokens[-50:])[-200:]
            yield TokenBatchEvent(tokens=accepted)
            delta = "".join(token.text for token in accepted).strip()
            if delta:
                yield TranscriptDeltaEvent(delta=delta)

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
        tokens: list[TranscriptToken] = []
        prompt_carry = prompt
        max_buffer = 0

        async for chunk in chunks:
            if not chunk:
                continue
            buffer.extend(chunk)
            if len(buffer) > max_buffer:
                max_buffer = len(buffer)

            while len(buffer) >= self.window_bytes:
                offset_seconds = consumed_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                window = bytes(buffer[: self.window_bytes])
                window_tokens = await asr.transcribe_pcm(
                    window,
                    language=language,
                    prompt=prompt_carry,
                )
                accepted = [
                    TranscriptToken(
                        start=token.start + offset_seconds,
                        end=token.end + offset_seconds,
                        text=token.text,
                        probability=token.probability,
                    )
                    for token in window_tokens
                ]
                if accepted:
                    tokens.extend(accepted)
                    prompt_carry = "".join(
                        token.text for token in tokens[-50:]
                    )[-200:]
                    yield TokenBatchEvent(tokens=accepted)
                    delta = "".join(token.text for token in accepted).strip()
                    if delta:
                        yield TranscriptDeltaEvent(delta=delta)

                del buffer[: self.step_bytes]
                consumed_bytes += self.step_bytes

        if buffer:
            offset_seconds = consumed_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            window = bytes(buffer)
            window_tokens = await asr.transcribe_pcm(
                window,
                language=language,
                prompt=prompt_carry,
            )
            accepted = [
                TranscriptToken(
                    start=token.start + offset_seconds,
                    end=token.end + offset_seconds,
                    text=token.text,
                    probability=token.probability,
                )
                for token in window_tokens
            ]
            if accepted:
                tokens.extend(accepted)
                prompt_carry = "".join(
                    token.text for token in tokens[-50:]
                )[-200:]
                yield TokenBatchEvent(tokens=accepted)
                delta = "".join(token.text for token in accepted).strip()
                if delta:
                    yield TranscriptDeltaEvent(delta=delta)

        self._stream_chunks_buffer_highwater = max_buffer
