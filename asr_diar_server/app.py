"""FastAPI application factory for asr_diar_server.

Module-level ``app`` is a lightweight instance created from default
``ServerSettings``. Heavy model initialisation happens in the lifespan,
not at import time.

Usage (ASGI):
    uvicorn asr_diar_server.app:app
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
        from asr_diar_server.backends.faster_whisper import build_asr_adapter
        from asr_diar_server.backends.nemo import build_diarization_adapter
        from asr_diar_server.pipelines.streaming import StreamingPipeline
        from asr_diar_server.pipelines.full_memory import FullMemoryPipeline

        application.state.settings = settings
        application.state.runtime = runtime

        # Build ASR adapter (always required)
        asr_adapter = build_asr_adapter(
            settings.model_asr,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
        runtime.asr_adapter = asr_adapter

        # Build optional diarization adapter
        diarization_adapter = None
        if settings.backend_diarization == "nemo" and settings.model_diarization:
            diarization_adapter = build_diarization_adapter(settings.model_diarization)
            runtime.diarization_adapter = diarization_adapter

            if settings.pipeline in ("chunked-file", "streaming"):
                from asr_diar_server.backends.nemo_streaming import StreamingDiarizerFactory

                streaming_factory = StreamingDiarizerFactory(
                    diarization_adapter._model, tier=settings.diarization_latency,
                )
                runtime.streaming_diarizer_factory = streaming_factory
                runtime.diarization_latency = settings.diarization_latency

        # Construct the pipeline
        if settings.pipeline in ("chunked-file", "streaming"):
            runtime.pipeline = StreamingPipeline(
                asr=asr_adapter,
                streaming_diarizer_factory=runtime.streaming_diarizer_factory,
            )
        else:
            runtime.pipeline = FullMemoryPipeline(asr=asr_adapter, diarization=diarization_adapter)

        # Server Warmup
        if settings.warmup == "enabled":
            from asr_diar_server.audio import AudioInput
            from asr_diar_server.bench.data import WARMUP_AUDIO_PATH

            warmup_audio = AudioInput(WARMUP_AUDIO_PATH.read_bytes())
            await runtime.pipeline.transcribe(warmup_audio)
            runtime.warmup_ready = True
        else:
            logger.warning("Server Warmup is disabled — first request may pay cold-model costs.")
            runtime.warmup_ready = True

        yield

        # Cleanup: adapters do not currently expose explicit teardown hooks.

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
