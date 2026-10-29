"""Pipeline Stream Event Types.

Internal and public SSE events emitted by transcription pipelines. The
``StreamEvent`` union covers all internal events; ``PipelineStreamEvent``
covers only the events exposed to SSE clients.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from coro.core.models.transcript import TranscriptToken


@dataclass
class TokenBatchEvent:
    """Internal event: a batch of accepted tokens from one ASR window.

    Never emitted to SSE clients; consumed internally by pipelines.
    """

    tokens: list[TranscriptToken]
    type: str = field(default="_tokens", init=False)


@dataclass
class TranscriptDeltaEvent:
    """Public SSE event: incremental transcript text delta."""

    delta: str
    type: str = field(default="transcript.text.delta", init=False)


@dataclass
class TranscriptDoneEvent:
    """Public SSE event: final transcript JSON string."""

    text: str
    type: str = field(default="transcript.text.done", init=False)


@runtime_checkable
class RenderedDoneFrame(Protocol):
    """Public SSE event: a done frame that renders its own SSE lines lazily.

    The Streaming Pipeline emits one in place of a ``TranscriptDoneEvent`` so the
    final transcript is read from its spill store instead of held in memory.
    """

    def iter_sse(self) -> Iterator[str]:
        """Yield the frame's SSE lines one at a time."""
        ...


# MARK: Stream Event Union
StreamEvent = TokenBatchEvent | TranscriptDeltaEvent | TranscriptDoneEvent
PipelineStreamEvent = TranscriptDeltaEvent | TranscriptDoneEvent | RenderedDoneFrame
