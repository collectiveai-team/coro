"""Boundary Response Schema models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# Item Models ---------------------------------------------------------------
class WhisperWord(BaseModel):
    """Word-level timestamp item in a segment."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    score: float
    speaker: str


class WhisperSegment(BaseModel):
    """Segment item in the Whisper Response Schema."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    speaker: str
    words: list[WhisperWord]


class WhisperTranscriptItem(BaseModel):
    """Transcript convenience item."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str


class WhisperDiarizationItem(BaseModel):
    """Diarization convenience item."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    speaker: str


class RawWord(BaseModel):
    """Raw ASR word item before segment interpolation."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    score: float


# Response Model ------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    """Strict internal transcription response schema exposed by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    segments: list[WhisperSegment]
    word_segments: list[WhisperWord]
    transcript: list[WhisperTranscriptItem]
    diarization: list[WhisperDiarizationItem]
    raw_words: list[RawWord]


# MARK: OpenAI-Style Transcription Response Schemas
class TranscriptionUsage(BaseModel):
    """OpenAI-style transcription usage object."""

    model_config = ConfigDict(extra="forbid")

    type: str
    seconds: int


class JsonResponse(BaseModel):
    """Default OpenAI-style JSON transcription response."""

    model_config = ConfigDict(extra="forbid")

    text: str
    usage: TranscriptionUsage


class VerboseJsonSegment(BaseModel):
    """Segment item in an OpenAI-style verbose JSON response."""

    model_config = ConfigDict(extra="forbid")

    id: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class VerboseJsonWord(BaseModel):
    """Word item in an OpenAI-style verbose JSON response."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float


class VerboseJsonResponse(BaseModel):
    """OpenAI-style verbose JSON transcription response."""

    model_config = ConfigDict(extra="forbid")

    duration: float
    language: str
    text: str
    segments: list[VerboseJsonSegment]
    words: list[VerboseJsonWord]
    usage: TranscriptionUsage


class DiarizadJsonSegment(BaseModel):
    """Speaker-annotated segment in a diarized JSON response."""

    model_config = ConfigDict(extra="forbid")

    type: str
    id: str
    start: float
    end: float
    text: str
    speaker: str


class DiarizadJsonResponse(BaseModel):
    """OpenAI-style diarized JSON transcription response."""

    model_config = ConfigDict(extra="forbid")

    task: str
    duration: float
    text: str
    segments: list[DiarizadJsonSegment]
    usage: TranscriptionUsage


DiarizedJsonResponse = DiarizadJsonResponse


# MARK: OpenAI-Style Error Schema
class OpenAIError(BaseModel):
    """OpenAI-style error object."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


# Error Response Model ------------------------------------------------------
class OpenAIErrorResponse(BaseModel):
    """OpenAI-style error response boundary schema."""

    error: OpenAIError

    @classmethod
    def from_error(
        cls,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> OpenAIErrorResponse:
        return cls(
            error=OpenAIError(
                message=message,
                type=error_type,
                param=param,
                code=code,
            )
        )
