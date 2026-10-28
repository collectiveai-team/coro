"""Core Boundary data models — Project-Owned Transcript Model tree.

API-agnostic dataclasses grouped by concern: transcript domain types,
the transcription response model, and pipeline stream events. These cross
package boundaries so backend-native types do not leak through; backend
adapters convert native objects into these types at adapter edges.
"""

from __future__ import annotations

from coro.core.models.events import (
    PipelineStreamEvent,
    RenderedDoneFrame,
    StreamEvent,
    TokenBatchEvent,
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
)
from coro.core.models.response import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
)
from coro.core.models.transcript import (
    SpeakerSegment,
    TranscriptToken,
    TranscriptWord,
)

__all__ = [
    "DiarizationItem",
    "PipelineStreamEvent",
    "RawWord",
    "RenderedDoneFrame",
    "ResponseSegment",
    "SpeakerSegment",
    "StreamEvent",
    "TokenBatchEvent",
    "TranscriptDeltaEvent",
    "TranscriptDoneEvent",
    "TranscriptItem",
    "TranscriptToken",
    "TranscriptWord",
    "TranscriptionResult",
]
