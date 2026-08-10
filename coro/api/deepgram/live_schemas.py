"""Deepgram-shaped live (WebSocket) messages.

Deepgram's streaming transport speaks a different vocabulary from its
pre-recorded response: the server pushes ``Results`` frames as audio is
transcribed, then a closing ``Metadata`` frame. Each frame carries one
channel's alternatives, and per-word speakers ride on the words exactly as
they do in the REST shape.

The same fidelity policy applies as for the REST endpoint — see ADR 0010.
Fields coro cannot measure are omitted, never stubbed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from coro.api.deepgram.schemas import DeepgramWord

RESULTS = "Results"
METADATA = "Metadata"

PRIMARY_CHANNEL_INDEX = [0, 1]
"""Deepgram reports ``[channel, total_channels]``; coro is always mono."""


class DeepgramLiveAlternative(BaseModel):
    """One hypothesis inside a live ``Results`` frame."""

    model_config = ConfigDict(extra="forbid")

    transcript: str
    confidence: float
    words: list[DeepgramWord]


class DeepgramLiveChannel(BaseModel):
    """The ``channel`` object of a live ``Results`` frame."""

    model_config = ConfigDict(extra="forbid")

    alternatives: list[DeepgramLiveAlternative]


class DeepgramLiveResults(BaseModel):
    """A ``Results`` frame.

    ``is_final`` marks a span the server will not revise. coro's windowing
    only surfaces tokens it has already accepted, so every frame it emits is
    final; ``interim_results`` is honoured by simply having no interim frames
    to suppress.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = RESULTS
    channel_index: list[int] = PRIMARY_CHANNEL_INDEX
    duration: float
    start: float
    is_final: bool
    speech_final: bool
    channel: DeepgramLiveChannel


class DeepgramLiveMetadata(BaseModel):
    """The closing ``Metadata`` frame."""

    model_config = ConfigDict(extra="forbid")

    type: str = METADATA
    request_id: str
    created: str
    duration: float
    channels: int
    models: list[str]


class DeepgramLiveError(BaseModel):
    """An error frame sent before closing the socket."""

    model_config = ConfigDict(extra="forbid")

    type: str = "Error"
    description: str
    message: str


def live_results(
    words: list[DeepgramWord],
    *,
    start: float,
    duration: float,
    is_final: bool = True,
    speech_final: bool = True,
) -> DeepgramLiveResults:
    """Build a ``Results`` frame from already-attributed words."""
    transcript = " ".join(word.word for word in words).strip()
    confidence = sum(word.confidence for word in words) / len(words) if words else 0.0
    return DeepgramLiveResults(
        duration=duration,
        start=start,
        is_final=is_final,
        speech_final=speech_final,
        channel=DeepgramLiveChannel(
            alternatives=[
                DeepgramLiveAlternative(transcript=transcript, confidence=confidence, words=words)
            ]
        ),
    )
