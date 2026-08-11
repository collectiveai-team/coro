"""Deepgram-shaped transcription response for the ``POST /v1/listen`` endpoint.

A documented subset of Deepgram's pre-recorded response (see ADR 0015),
carrying per-word speaker labels. Served from Deepgram's own endpoint rather
than from a ``response_format`` value, so the OpenAI-compatible surface is
never extended with values OpenAI does not define.

Timestamps are floating-point seconds, which is Deepgram's unit.

``speaker_confidence`` is deliberately **omitted** — coro's diarization
adapters binarize their per-frame posteriors into a speaker timeline before it
reaches the Core Boundary, so the quantity Deepgram names is not available.
See ADR 0015.

``confidence`` is omitted by the same rule whenever the ASR backend expresses
no probability. Deepgram types it ``Optional[float]`` at word, alternative and
utterance level, so absence is valid against the vendor's own SDK types, and it
is the only honest encoding: a stubbed ``1.0`` is indistinguishable from a
measured certainty on the wire.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from coro.api.schemas import TranscriptionResponse, TranscriptWord
from coro.api.utterances import (
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
    """Word item in a Deepgram-shaped response.

    ``confidence`` is absent when the ASR backend reported no probability.
    """

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    confidence: float | None = None
    speaker: int | None = None


class DeepgramAlternative(BaseModel):
    """A single transcription hypothesis for one channel.

    ``confidence`` is the mean over the words that carry one, and absent when
    none of them do.
    """

    model_config = ConfigDict(extra="forbid")

    transcript: str
    confidence: float | None = None
    words: list[DeepgramWord]


class DeepgramChannel(BaseModel):
    """One audio channel's hypotheses."""

    model_config = ConfigDict(extra="forbid")

    alternatives: list[DeepgramAlternative]


class DeepgramUtterance(BaseModel):
    """A maximal same-speaker word run, with its words retained.

    ``confidence`` is the mean over the words that carry one, and absent when
    none of them do.
    """

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    confidence: float | None = None
    channel: int
    transcript: str
    words: list[DeepgramWord]
    speaker: int | None = None


class DeepgramResults(BaseModel):
    """The ``results`` object of a Deepgram-shaped response.

    ``utterances`` is present only when the request asked for it, matching
    Deepgram, which gates the speaker-turn view behind ``utterances=true``.
    """

    model_config = ConfigDict(extra="forbid")

    channels: list[DeepgramChannel]
    utterances: list[DeepgramUtterance] | None = None


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


class DeepgramErrorResponse(BaseModel):
    """Deepgram-shaped error body.

    Deepgram's endpoint reports failures as ``err_code``/``err_msg``, not as an
    OpenAI ``error`` object, so ``/v1/listen`` must not reuse the app-wide
    OpenAI-style handler.
    """

    model_config = ConfigDict(extra="forbid")

    err_code: str
    err_msg: str
    request_id: str


def _word(word: TranscriptWord, *, diarize: bool) -> DeepgramWord:
    return DeepgramWord(
        word=word.word,
        start=word.start,
        end=word.end,
        confidence=word.score,
        speaker=_speaker_label(word.speaker) if diarize else None,
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
    diarize: bool = True,
    utterances: bool = True,
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
        diarize: Whether the request asked for speaker labels. When false, no
            word carries a speaker, matching Deepgram's default.
        utterances: Whether the request asked for the speaker-turn view.

    Returns:
        The Deepgram-shaped response. Fields left as ``None`` are omitted at
        serialization, so an undiarized response has no ``speaker`` keys rather
        than null ones — Deepgram never emits a null speaker.

    """
    words = [_word(word, diarize=diarize) for word in result.word_segments]
    turns = (
        [
            DeepgramUtterance(
                start=utterance.start,
                end=utterance.end,
                confidence=utterance.confidence,
                channel=PRIMARY_CHANNEL,
                transcript=utterance.text,
                words=[_word(word, diarize=diarize) for word in utterance.words],
                speaker=_speaker_label(utterance.speaker) if diarize else None,
            )
            for utterance in group_words_into_utterances(result.word_segments)
        ]
        if utterances
        else None
    )
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
            utterances=turns,
        ),
    )
