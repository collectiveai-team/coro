"""v2 transcription router — /v2/audio/transcriptions.

Uses the disk-backed Transcription Pipeline.  Accepts the same
OpenAI-Compatible Request fields as v1.  The route handler spools the
upload to a temp file and delegates to the v2_pipeline in RuntimeState.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from asr_diar_server.api.errors import openai_error
from asr_diar_server.api.sse import streaming_response

router = APIRouter(prefix="/v2")

_JSON_LIKE_FORMATS = {
    None,
    "",
    "json",
    "verbose_json",
    "verbose-json",
    "diarized_json",
    "diarized-json",
}
_UNSUPPORTED_FORMATS = {"text", "srt", "vtt", "tsv"}


def _unlink(path: str) -> None:
    """Remove a file silently if it exists."""
    with contextlib.suppress(FileNotFoundError):
        Path(path).unlink()


@router.post("/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="", description="Accepted but ignored."),
    language: str | None = Form(default=None),
    prompt: str = Form(default=""),
    response_format: str | None = Form(default=None),
    stream: bool = Form(default=False),
) -> Response:
    """Accept audio and return a WhisperX-Style Response via disk-backed pipeline.

    Uploads are spooled to a temporary file so the pipeline can stream PCM
    chunks without keeping the full audio in memory.
    """
    if response_format and response_format.lower() not in _JSON_LIKE_FORMATS:
        if response_format.lower() in _UNSUPPORTED_FORMATS:
            return openai_error(
                f"response_format '{response_format}' is not supported.",
                param="response_format",
            )
        return openai_error(
            f"Unknown response_format '{response_format}'.", param="response_format"
        )

    # Spool upload to a temp file; check for empty content.
    fd, path = tempfile.mkstemp(prefix="asr-upload-", suffix=".audio")
    wrote_data = False
    try:
        with os.fdopen(fd, "wb") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                wrote_data = True
                tmp.write(chunk)
    except Exception:
        _unlink(path)
        raise

    if not wrote_data:
        _unlink(path)
        return openai_error("Empty audio file.", param="file")

    runtime = request.app.state.runtime
    pipeline = getattr(runtime, "v2_pipeline", None)
    if pipeline is None:
        _unlink(path)
        return openai_error(
            "Server is not ready. No v2 pipeline is available.",
            error_type="server_error",
            status_code=503,
        )

    if stream:
        stream_method = getattr(pipeline, "stream_from_path", None)
        if stream_method is None:
            _unlink(path)
            return openai_error(
                "v2 pipeline does not support streaming.",
                error_type="server_error",
                status_code=503,
            )

        async def _cleanup_after_stream():
            try:
                async for event in stream_method(path, language=language, prompt=prompt or None):
                    yield event
            finally:
                _unlink(path)

        return streaming_response(_cleanup_after_stream())

    try:
        result = await pipeline.run_from_path(path, language=language, prompt=prompt or None)
    finally:
        _unlink(path)

    return JSONResponse(result)
