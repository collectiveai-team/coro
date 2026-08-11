"""Typed transcription exceptions for API boundary handling."""

from __future__ import annotations

from fastapi import status

# MARK: Client-facing Messages
UNDECODABLE_MEDIA_MESSAGE = (
    "Could not decode the uploaded file as audio or video. "
    "Ensure it is a supported, non-corrupt media format."
)
"""Safe, ffmpeg-detail-free message for an upload that cannot be decoded."""


# MARK: Base Transcription Exception
class TranscriptionError(Exception):
    """Base exception translated to an OpenAI-style error response."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "server_error"

    def __init__(self, message: str, *, param: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


# MARK: Public API Failure Types
class TranscriptionValidationError(TranscriptionError):
    """Request validation failure."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "invalid_request_error"


class TranscriptionReadinessError(TranscriptionError):
    """Server readiness failure."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "server_error"


class UnsupportedStreamingError(TranscriptionError):
    """Configured pipeline does not support streaming."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "server_error"


class TranscriptionProcessingError(TranscriptionError):
    """Audio conversion, ASR, diarization, or response assembly failure."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "server_error"


class TranscriptionCapacityError(TranscriptionError):
    """Admission-control rejection: the ASR queue-depth cap was exceeded.

    Carries ``retry_after_seconds``, which the exception handler renders as a
    ``Retry-After`` response header alongside the OpenAI-Style Error body.
    """

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_type = "rate_limit_exceeded"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float,
        param: str | None = None,
        code: str | None = "server_overloaded",
    ) -> None:
        super().__init__(message, param=param, code=code)
        self.retry_after_seconds = retry_after_seconds
