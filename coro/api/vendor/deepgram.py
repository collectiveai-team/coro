"""Deepgram-shaped transcription response.

A documented subset of Deepgram's ``/v1/listen`` pre-recorded response (see
ADR 0010), carrying per-word speaker labels. Emitted only for
``response_format=deepgram_json``; the OpenAI formats are untouched.

Timestamps are floating-point seconds, which is Deepgram's unit.

``speaker_confidence`` is deliberately **omitted** — coro's diarization
adapters binarize their per-frame posteriors into a speaker timeline before it
reaches the Core Boundary, so the quantity Deepgram names is not available.
See ADR 0010.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from coro.api.schemas import TranscriptionResponse, WhisperWord
from coro.api.vendor.utterances import (
    UNKNOWN_SPEAKER_LABEL,
    group_words_into_utterances,
    mean_confidence,
)

MONO_CHANNEL_COUNT = 1
"""Uploads are converted to mono before transcription."""

PRIMARY_CHANNEL = 0
"""Deepgram channel index for the single audio channel coro produces."""


def _speaker_label(speaker: str) -> int | None:
    """Map a coro speaker label onto Deepgram's optional integer speaker field.

    coro's ``-1`` sentinel means the diarization timeline does not support the
    word; Deepgram spells that absence as ``null``. Speaker numbering is passed
    through rather than renumbered, so labels stay comparable with the
    ``diarized_json`` projection and with the STM used for quality scoring.
    """
    if speaker == UNKNOWN_SPEAKER_LABEL:
        return None
    try:
        return int(speaker)
    except ValueError:
        return None


class DeepgramWord(BaseModel):
    """Word item in a Deepgram-shaped response."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    confidence: float
    speaker: int | None = None


class DeepgramAlternative(BaseModel):
    """A single transcription hypothesis for one channel."""

    model_config = ConfigDict(extra="forbid")

    transcript: str
    confidence: float
    words: list[DeepgramWord]


class DeepgramChannel(BaseModel):
    """One audio channel's hypotheses."""

    model_config = ConfigDict(extra="forbid")

    alternatives: list[DeepgramAlternative]


class DeepgramUtterance(BaseModel):
    """A maximal same-speaker word run, with its words retained."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    confidence: float
    channel: int
    transcript: str
    words: list[DeepgramWord]
    speaker: int | None = None


class DeepgramResults(BaseModel):
    """The ``results`` object of a Deepgram-shaped response."""

    model_config = ConfigDict(extra="forbid")

    channels: list[DeepgramChannel]
    utterances: list[DeepgramUtterance]


class DeepgramMetadata(BaseModel):
    """The ``metadata`` object of a Deepgram-shaped response."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    sha256: str
    created: str
    duration: float
    channels: int
    models: list[str]
    model_info: dict[str, Any]


class DeepgramResponse(BaseModel):
    """Deepgram-shaped transcription response (documented subset)."""

    model_config = ConfigDict(extra="forbid")

    metadata: DeepgramMetadata
    results: DeepgramResults


def _word(word: WhisperWord) -> DeepgramWord:
    return DeepgramWord(
        word=word.word,
        start=word.start,
        end=word.end,
        confidence=word.score,
        speaker=_speaker_label(word.speaker),
    )


def deepgram_response(
    result: TranscriptionResponse,
    *,
    text: str,
    duration: float,
    request_id: str,
    audio_sha256: str,
    created: str,
    asr_model: str,
    asr_backend: str,
) -> DeepgramResponse:
    """Project the internal result onto Deepgram's pre-recorded response shape.

    Args:
        result: The validated Strict Transcription Response Schema instance.
        text: The full transcript text, already assembled by the caller.
        duration: Audio duration in seconds.
        request_id: The server's request id.
        audio_sha256: Hex SHA-256 of the uploaded audio bytes.
        created: ISO 8601 completion timestamp.
        asr_model: The configured ASR Model Selection.
        asr_backend: The configured ASR Backend Provider.

    Returns:
        The Deepgram-shaped response.

    """
    words = [_word(word) for word in result.word_segments]
    utterances = [
        DeepgramUtterance(
            start=utterance.start,
            end=utterance.end,
            confidence=utterance.confidence,
            channel=PRIMARY_CHANNEL,
            transcript=utterance.text,
            words=[_word(word) for word in utterance.words],
            speaker=_speaker_label(utterance.speaker),
        )
        for utterance in group_words_into_utterances(result.word_segments)
    ]
    return DeepgramResponse(
        metadata=DeepgramMetadata(
            request_id=request_id,
            sha256=audio_sha256,
            created=created,
            duration=duration,
            channels=MONO_CHANNEL_COUNT,
            models=[asr_model],
            model_info={asr_model: {"name": asr_model, "arch": asr_backend}},
        ),
        results=DeepgramResults(
            channels=[
                DeepgramChannel(
                    alternatives=[
                        DeepgramAlternative(
                            transcript=text,
                            confidence=mean_confidence(result.word_segments),
                            words=words,
                        )
                    ]
                )
            ],
            utterances=utterances,
        ),
    )
