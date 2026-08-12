"""OpenAI-style error response helpers.

Transcription endpoints return OpenAI-style error objects so that
OpenAI-compatible clients can parse failures consistently.
"""

from __future__ import annotations

import math

from fastapi import Request
from fastapi.responses import JSONResponse

from coro.api.exceptions import TranscriptionError
from coro.api.openai.schemas import OpenAIErrorResponse


def openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
    status_code: int = 400,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return a JSONResponse shaped as an OpenAI-style error.

    Args:
        message: Human-readable error description.
        error_type: OpenAI error type string.
        param: The request parameter that caused the error.
        code: Optional machine-readable error code.
        status_code: HTTP status code.
        headers: Optional extra response headers (e.g. ``Retry-After``).

    Returns:
        JSONResponse with ``{"error": {...}}`` body.

    """
    body = OpenAIErrorResponse.from_error(
        message=message,
        error_type=error_type,
        param=param,
        code=code,
    )
    return JSONResponse(body.model_dump(), status_code=status_code, headers=headers)


async def transcription_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Translate typed transcription failures to OpenAI-style errors.

    Failures carrying a ``retry_after_seconds`` retry hint (admission-control
    rejections) additionally get a ``Retry-After`` response header.

    Registered only for ``TranscriptionError``; the broad ``Exception`` type
    matches Starlette's ``add_exception_handler`` handler signature.
    """
    assert isinstance(exc, TranscriptionError)  # noqa: S101
    headers: dict[str, str] = {}
    retry_after = getattr(exc, "retry_after_seconds", None)
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
    return openai_error(
        exc.message,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
        status_code=exc.status_code,
        headers=headers or None,
    )
