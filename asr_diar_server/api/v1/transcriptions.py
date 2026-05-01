"""Transcription Endpoint router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns a WhisperX-Style Response.
The route handler stays thin; orchestration delegates to the configured pipeline.
"""

from __future__ import annotations

from enum import StrEnum

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from asr_diar_server.api.dependencies import get_pipeline
from asr_diar_server.api.exceptions import (
    TranscriptionProcessingError,
    TranscriptionValidationError,
    UnsupportedStreamingError,
)
from asr_diar_server.api.schemas import WhisperXResponse
from asr_diar_server.api.sse import streaming_response
from asr_diar_server.audio import AudioInput


# MARK: Router Configuration
router = APIRouter(prefix="/v1")


# MARK: Response Format Enum
class ResponseFormat(StrEnum):
    """All OpenAI response_format values this server recognises."""

    # JSON-like: all map to the WhisperX-Style Response
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    VERBOSE_JSON_HYPHEN = "verbose-json"
    DIARIZED_JSON = "diarized_json"
    DIARIZED_JSON_HYPHEN = "diarized-json"

    # Unsupported OpenAI formats (recognised but not implemented)
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


# Response Formats ----------------------------------------------------------
# Formats that map to the WhisperX-Style JSON Response.
_JSON_LIKE_FORMATS = {
    ResponseFormat.JSON,
    ResponseFormat.VERBOSE_JSON,
    ResponseFormat.VERBOSE_JSON_HYPHEN,
    ResponseFormat.DIARIZED_JSON,
    ResponseFormat.DIARIZED_JSON_HYPHEN,
}

# Formats that are valid OpenAI values but not supported here.
_UNSUPPORTED_FORMATS = {
    ResponseFormat.TEXT,
    ResponseFormat.SRT,
    ResponseFormat.VTT,
    ResponseFormat.TSV,
}


# MARK: Transcription Endpoint
@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(
        default="", description="Accepted but ignored; server uses configured backend."
    ),
    language: str | None = Form(default=None, description="Optional BCP-47 language hint."),
    prompt: str = Form(default="", description="Optional initial prompt for transcription."),
    response_format: str | None = Form(default=None, description="Response format."),
    stream: bool = Form(default=False, description="If true, return OpenAI-Exact SSE."),
    pipeline=Depends(get_pipeline),
) -> Response:
    """Accept audio and return a WhisperX-Style Response.

    Supported response formats: json, verbose_json, diarized_json (and empty).
    All map to the same enriched WhisperX-Style Response.
    """
    # Request Validation ----------------------------------------------------
    if response_format:
        try:
            fmt = ResponseFormat(response_format.lower())
        except ValueError:
            raise TranscriptionValidationError(
                f"Unknown response_format '{response_format}'.",
                param="response_format",
            ) from None
        if fmt in _UNSUPPORTED_FORMATS:
            raise TranscriptionValidationError(
                f"response_format '{response_format}' is not supported. "
                "Supported formats: json, verbose_json.",
                param="response_format",
            )

    audio = await AudioInput.from_upload(file)
    if not await audio.read_bytes():
        raise TranscriptionValidationError("Empty audio file.", param="file")

    # Streaming Response ----------------------------------------------------
    if stream:
        stream_method = getattr(pipeline, "stream", None)
        if stream_method is None:
            raise UnsupportedStreamingError("Configured pipeline does not support streaming.")
        return streaming_response(stream_method(audio, language=language, prompt=prompt or None))

    # JSON Response ---------------------------------------------------------
    try:
        result = await pipeline.transcribe(audio, language=language, prompt=prompt or None)
    except TranscriptionValidationError:
        raise
    except Exception as exc:
        raise TranscriptionProcessingError("Transcription processing failed.") from exc
    return JSONResponse(WhisperXResponse.model_validate(result).model_dump())
