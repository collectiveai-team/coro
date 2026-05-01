"""v1 transcription router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns a WhisperX-Style
Response.  The route handler stays thin; orchestration delegates to the
v1_pipeline injected in RuntimeState.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from asr_diar_server.api.errors import openai_error
from asr_diar_server.api.sse import streaming_response

router = APIRouter(prefix="/v1")

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


@router.post("/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(
        default="", description="Accepted but ignored; server uses configured backend."
    ),
    language: str | None = Form(default=None, description="Optional BCP-47 language hint."),
    prompt: str = Form(default="", description="Optional initial prompt for transcription."),
    response_format: str | None = Form(default=None, description="Response format."),
    stream: bool = Form(default=False, description="If true, return OpenAI-Exact SSE."),
) -> Response:
    """Accept audio and return a WhisperX-Style Response.

    Supported response formats: json, verbose_json, diarized_json (and empty).
    All map to the same enriched WhisperX-Style Response.
    """
    # Validate response_format early.
    if response_format and response_format.lower() not in _JSON_LIKE_FORMATS:
        if response_format.lower() in _UNSUPPORTED_FORMATS:
            return openai_error(
                f"response_format '{response_format}' is not supported. "
                "Supported formats: json, verbose_json.",
                param="response_format",
            )
        return openai_error(
            f"Unknown response_format '{response_format}'.",
            param="response_format",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        return openai_error("Empty audio file.", param="file")

    runtime = request.app.state.runtime
    pipeline = getattr(runtime, "v1_pipeline", None)
    if pipeline is None:
        return openai_error(
            "Server is not ready. No v1 pipeline is available.",
            error_type="server_error",
            status_code=503,
        )

    if stream:
        stream_method = getattr(pipeline, "stream", None)
        if stream_method is None:
            return openai_error(
                "v1 pipeline does not support streaming.",
                error_type="server_error",
                status_code=503,
            )
        return streaming_response(
            stream_method(audio_bytes, language=language, prompt=prompt or None)
        )

    result = await pipeline.run(audio_bytes, language=language, prompt=prompt or None)
    return JSONResponse(result)
