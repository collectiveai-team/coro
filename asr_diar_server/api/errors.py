"""OpenAI-style error response helpers.

Transcription endpoints return OpenAI-style error objects so that
OpenAI-compatible clients can parse failures consistently.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse


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
    body: dict = {"message": message, "type": error_type}
    if param is not None:
        body["param"] = param
    if code is not None:
        body["code"] = code
    return JSONResponse({"error": body}, status_code=status_code)
