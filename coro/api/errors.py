"""OpenAI-style error response helpers.

Transcription endpoints return OpenAI-style error objects so that
OpenAI-compatible clients can parse failures consistently.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from coro.api.exceptions import TranscriptionError
from coro.api.schemas import OpenAIErrorResponse


def openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
    status_code: int = 400,
) -> JSONResponse:
    """Return a JSONResponse shaped as an OpenAI-style error.

    Args:
        message: Human-readable error description.
        error_type: OpenAI error type string.
        param: The request parameter that caused the error.
        code: Optional machine-readable error code.
        status_code: HTTP status code.

    Returns:
        JSONResponse with ``{"error": {...}}`` body.

    """
    body = OpenAIErrorResponse.from_error(
        message=message,
        error_type=error_type,
        param=param,
        code=code,
    )
    return JSONResponse(body.model_dump(), status_code=status_code)


async def transcription_exception_handler(_request, exc: TranscriptionError) -> JSONResponse:
    """Translate typed transcription failures to OpenAI-style errors."""
    return openai_error(
        exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
        status_code=exc.status_code,
    )
