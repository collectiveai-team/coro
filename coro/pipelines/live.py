"""Live transcription over an open socket.

The Streaming Pipeline already consumes PCM incrementally — ``ASRWindowing``
takes any async iterator of chunks, and a StreamingDiarizer ingests chunk by
chunk. It reads from a spooled file only because an HTTP upload arrives whole.

This module supplies the other source: audio that is still being produced. The
transcription machinery below the source is identical, so a socket stream and
an upload cannot drift apart in behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.core.models import SpeakerSegment, TokenBatchEvent, TranscriptToken
from coro.core.protocols import ASRAdapter
from coro.pipelines.windowing import ASRWindowing

logger = logging.getLogger(__name__)

_SENTINEL = object()


class LiveAudioSource:
    """An async PCM chunk iterator fed by a producer that is still running.

    The socket handler pushes frames in as they arrive and calls
    :meth:`close` when the client signals end of stream; the windowing layer
    pulls from the other end and cannot tell the difference from a file.
    """

    def __init__(self, *, max_pending_chunks: int = 64) -> None:
        # Bounded so a client that floods audio faster than the ASR consumes it
        # applies backpressure instead of growing the queue without limit.
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending_chunks)
        self._closed = False

    async def push(self, chunk: bytes) -> None:
        """Hand one PCM chunk to the consumer, waiting if it is behind."""
        if self._closed or not chunk:
            return
        await self._queue.put(chunk)

    async def close(self) -> None:
        """Signal end of stream; the consumer finishes its current work."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(_SENTINEL)

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks until the producer closes the stream."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item


class LiveTranscriptionSession:
    """Drive ASR windowing and optional diarization over a live PCM stream.

    Tokens are surfaced as each ASR window completes, so a caller can emit
    incremental results. The speaker timeline is only meaningful once the
    diarizer has seen the whole stream, so :meth:`finalize` yields it at the
    end — per-word speakers therefore attach to final results, not interim
    ones.
    """

    def __init__(
        self,
        *,
        asr: ASRAdapter,
        windowing: ASRWindowing | None = None,
        streaming_diarizer_factory=None,
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self._asr = asr
        self._windowing = windowing or ASRWindowing()
        self._language = language
        self._prompt = prompt
        self._diarizer = (
            streaming_diarizer_factory() if streaming_diarizer_factory is not None else None
        )
        self._total_bytes = 0
        self._started = time.perf_counter()

    @property
    def diarization_enabled(self) -> bool:
        """True when a streaming diarizer is attached to this session."""
        return self._diarizer is not None

    @property
    def audio_seconds(self) -> float:
        """Seconds of audio ingested so far, derived from bytes consumed."""
        return self._total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    async def run(self, source: LiveAudioSource) -> AsyncIterator[list[TranscriptToken]]:
        """Consume the source, yielding each window's accepted tokens."""

        async def _tee() -> AsyncIterator[bytes]:
            async for chunk in source.chunks():
                self._total_bytes += len(chunk)
                if self._diarizer is not None:
                    self._diarizer.ingest_pcm_chunk(chunk)
                yield chunk

        async for event in self._windowing.stream_chunks(
            _tee(),
            asr=self._asr,
            language=self._language,
            prompt=self._prompt,
        ):
            if isinstance(event, TokenBatchEvent) and event.tokens:
                yield event.tokens

    def finalize(self) -> list[SpeakerSegment]:
        """Return the completed speaker timeline, or empty without a diarizer."""
        if self._diarizer is None:
            return []
        timeline = self._diarizer.finalize()
        logger.info(
            "live_session finalize elapsed=%.3fs audio_s=%.2f timeline=%d",
            time.perf_counter() - self._started,
            self.audio_seconds,
            len(timeline),
        )
        return timeline
