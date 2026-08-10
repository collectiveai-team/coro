"""AssemblyAI-shaped transcription response.

A documented subset of AssemblyAI's transcript object (see ADR 0010), carrying
the per-word speaker labels that no OpenAI-compatible format has a slot for.
Emitted only for ``response_format=assemblyai_json``; the OpenAI formats are
untouched.

Timestamps are integer milliseconds, which is AssemblyAI's unit.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from coro.api.schemas import TranscriptionResponse, WhisperWord
from coro.api.vendor.utterances import (
    UNKNOWN_SPEAKER_LABEL,
    group_words_into_utterances,
    mean_confidence,
    seconds_to_milliseconds,
)

TRANSCRIPT_COMPLETED = "completed"
"""The only ``status`` a synchronous response can carry."""


def _speaker_label(speaker: str) -> str | None:
    """Map a coro speaker label onto AssemblyAI's optional speaker field.

    coro's ``-1`` sentinel means the diarization timeline does not support the
    word. AssemblyAI spells that absence as ``null``, so it is emitted as
    ``None`` rather than as the literal string ``"-1"``, which a client would
    otherwise read as a speaker *named* ``-1``.
    """
    return None if speaker == UNKNOWN_SPEAKER_LABEL else speaker


class AssemblyAIWord(BaseModel):
    """Word item in an AssemblyAI-shaped response."""

    model_config = ConfigDict(extra="forbid")

    text: str
    start: int
    end: int
    confidence: float
    speaker: str | None = None


class AssemblyAIUtterance(BaseModel):
    """A maximal same-speaker word run, with its words retained."""

    model_config = ConfigDict(extra="forbid")

    text: str
    start: int
    end: int
    confidence: float
    speaker: str | None = None
    words: list[AssemblyAIWord]


class AssemblyAIResponse(BaseModel):
    """AssemblyAI-shaped transcription response (documented subset)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    audio_url: str
    language_code: str | None = None
    audio_duration: int
    confidence: float
    text: str
    words: list[AssemblyAIWord]
    utterances: list[AssemblyAIUtterance]


def _word(word: WhisperWord) -> AssemblyAIWord:
    return AssemblyAIWord(
        text=word.word,
        start=seconds_to_milliseconds(word.start),
        end=seconds_to_milliseconds(word.end),
        confidence=word.score,
        speaker=_speaker_label(word.speaker),
    )


def assemblyai_response(
    result: TranscriptionResponse,
    *,
    text: str,
    duration: float,
    language: str | None,
    request_id: str,
    audio_url: str,
) -> AssemblyAIResponse:
    """Project the internal result onto AssemblyAI's transcript shape.

    Args:
        result: The validated Strict Transcription Response Schema instance.
        text: The full transcript text, already assembled by the caller.
        duration: Audio duration in seconds.
        language: The validated BCP-47 language hint, or None.
        request_id: The server's request id, surfaced as the transcript ``id``.
        audio_url: Provenance label for the submitted audio. This endpoint
            accepts uploaded bytes, so it is the upload filename rather than a
            dereferenceable URL (ADR 0010).

    Returns:
        The AssemblyAI-shaped response.

    """
    words = list(result.word_segments)
    utterances = [
        AssemblyAIUtterance(
            text=utterance.text,
            start=seconds_to_milliseconds(utterance.start),
            end=seconds_to_milliseconds(utterance.end),
            confidence=utterance.confidence,
            speaker=_speaker_label(utterance.speaker),
            words=[_word(word) for word in utterance.words],
        )
        for utterance in group_words_into_utterances(words)
    ]
    return AssemblyAIResponse(
        id=request_id,
        status=TRANSCRIPT_COMPLETED,
        audio_url=audio_url,
        language_code=language,
        audio_duration=math.ceil(duration),
        confidence=mean_confidence(words),
        text=text,
        words=[_word(word) for word in words],
        utterances=utterances,
    )
