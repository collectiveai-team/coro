"""OpenAI-shaped boundary schemas for ``POST /v1/audio/transcriptions``.

Mirrors ``openai.types.audio`` exactly; ``DiarizadJsonResponse`` is a
byte-exact clone of ``TranscriptionDiarized``. Conformance is asserted in
``tests/test_openai_sdk_conformance.py`` and the bytes are frozen in
``tests/test_openai_formats_unchanged.py``.

``OpenAIError`` is also the app-wide default error body, a legacy of coro
being OpenAI-first. A provider endpoint that needs its own error shape must
render it itself rather than raising ``TranscriptionError`` — see
``coro/api/deepgram/listen.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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
    seek: int
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
