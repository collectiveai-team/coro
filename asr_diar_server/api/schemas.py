"""Boundary Response Schema models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


# MARK: Strict WhisperX Response Schema
# Item Models ---------------------------------------------------------------
class WhisperXWord(BaseModel):
    """Word-level timestamp item in a segment."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    score: float
    speaker: str


class WhisperXSegment(BaseModel):
    """Segment item in the Strict WhisperX Response Schema."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    speaker: str
    words: list[WhisperXWord]


class WhisperXTranscriptItem(BaseModel):
    """Transcript convenience item."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str


class WhisperXDiarizationItem(BaseModel):
    """Diarization convenience item."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    speaker: str


class WhisperXRawWord(BaseModel):
    """Raw ASR word item before segment interpolation."""

    model_config = ConfigDict(extra="forbid")

    word: str
    start: float
    end: float
    score: float


# Response Model ------------------------------------------------------------
class WhisperXResponse(BaseModel):
    """Strict WhisperX Response Schema exposed by the transcription endpoint."""

    model_config = ConfigDict(extra="forbid")

    segments: list[WhisperXSegment]
    word_segments: list[WhisperXWord]
    transcript: list[WhisperXTranscriptItem]
    diarization: list[WhisperXDiarizationItem]
    raw_words: list[WhisperXRawWord]


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
