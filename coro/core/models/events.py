"""Pipeline Stream Event Types.

Internal and public SSE events emitted by transcription pipelines. The
``StreamEvent`` union covers all internal events; ``PipelineStreamEvent``
covers only the events exposed to SSE clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


# MARK: Stream Event Union
StreamEvent = TokenBatchEvent | TranscriptDeltaEvent | TranscriptDoneEvent
PipelineStreamEvent = TranscriptDeltaEvent | TranscriptDoneEvent
