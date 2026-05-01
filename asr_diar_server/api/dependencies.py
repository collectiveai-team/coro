"""FastAPI dependencies for settings and configured pipeline."""

from __future__ import annotations

from fastapi import Request

from asr_diar_server.api.exceptions import TranscriptionReadinessError
from asr_diar_server.settings import ServerSettings


# MARK: FastAPI Dependencies
def get_settings(request: Request) -> ServerSettings:
    """Return validated Server Startup Selection from app state."""
    return request.app.state.settings


def get_pipeline(request: Request):
    """Return the Singleton Runtime configured pipeline."""
    pipeline = getattr(request.app.state.runtime, "pipeline", None)
    if pipeline is None:
        raise TranscriptionReadinessError("Server is not ready. No pipeline is available.")
    return pipeline
