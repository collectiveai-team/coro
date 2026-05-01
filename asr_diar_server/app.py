"""FastAPI application factory for asr_diar_server.

Module-level ``app`` is a lightweight instance created from default
``ServerSettings``. Heavy model initialisation happens in the lifespan,
not at import time.

Usage (ASGI):
    uvicorn asr_diar_server.app:app
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asr_diar_server.api.errors import transcription_exception_handler
from asr_diar_server.api.exceptions import TranscriptionError
from asr_diar_server.runtime import RuntimeState
from asr_diar_server.settings import ServerSettings

if TYPE_CHECKING:
    pass


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create and configure a FastAPI application.

    Args:
        settings: Server settings. Defaults to ``ServerSettings()``.

    Returns:
        Configured FastAPI instance with no real model loaded.

    """
    if settings is None:
        settings = ServerSettings()

    runtime = RuntimeState(
        pipeline_selector=settings.pipeline,
        asr_provider=settings.backend_asr,
        asr_model=settings.model_asr,
        diarization_provider=settings.backend_diarization,
        diarization_model=settings.model_diarization,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Heavy model setup (TranscriptionEngine etc.) would go here in
        # production use.  Tests inject fake state via app.state.runtime.
        application.state.settings = settings
        application.state.runtime = runtime
        yield

    application = FastAPI(title="ASR Diarization Server", lifespan=lifespan)
    application.add_exception_handler(TranscriptionError, transcription_exception_handler)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from asr_diar_server.api.health import router as health_router
    from asr_diar_server.api.v1.transcriptions import router as v1_router

    application.state.settings = settings
    application.state.runtime = runtime
    application.include_router(health_router)
    application.include_router(v1_router)

    return application


# Lightweight module-level app: default settings, no model loaded.
# Standard ASGI launch: uvicorn asr_diar_server.app:app
app = create_app()
