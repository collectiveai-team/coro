"""Transcription Endpoint router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns a WhisperX-Style Response.
The route handler stays thin; orchestration delegates to the configured pipeline.
"""

from __future__ import annotations

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


# Response Formats ----------------------------------------------------------
# Response formats treated as aliases for the WhisperX-Style Response.
_JSON_LIKE_FORMATS = {
    None,
    "",
    "json",
    "verbose_json",
    "verbose-json",
    "diarized_json",
    "diarized-json",
}

# Unsupported formats: text, srt, vtt, etc.
# (Checked in Cycle 5; kept here for single definition.)
_UNSUPPORTED_FORMATS = {"text", "srt", "vtt", "tsv"}


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
    if response_format and response_format.lower() not in _JSON_LIKE_FORMATS:
        if response_format.lower() in _UNSUPPORTED_FORMATS:
            raise TranscriptionValidationError(
                f"response_format '{response_format}' is not supported. "
                "Supported formats: json, verbose_json.",
                param="response_format",
            )
        raise TranscriptionValidationError(
            f"Unknown response_format '{response_format}'.",
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
