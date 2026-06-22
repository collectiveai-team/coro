"""FastAPI application factory for coro.

Module-level ``app`` is a lightweight instance created from default
``ServerSettings``. Heavy model initialisation happens in the lifespan,
not at import time.

Usage (ASGI):
    uvicorn coro.app:app
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import coro

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coro.api.errors import transcription_exception_handler
from coro.api.exceptions import TranscriptionError
from coro.runtime import RuntimeState
from coro.settings import ServerSettings

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
        from coro.backends.diarization.nemo.diarization import build_diarization_adapter
        from coro.pipelines.streaming import StreamingPipeline
        from coro.pipelines.full_memory import FullMemoryPipeline

        application.state.settings = settings
        application.state.runtime = runtime
        logging.getLogger("coro").setLevel(settings.log_level.upper())
        logger.warning(
            "coro startup package_file=%s app_file=%s settings=%s",
            getattr(coro, "__file__", None),
            __file__,
            settings.model_dump(mode="json"),
        )

        # Build ASR adapter (always required), dispatching on the configured backend.
        if settings.backend_asr == "onnx-asr":
            from coro.backends.asr.onnx_asr import build_onnx_asr_adapter

            asr_adapter = build_onnx_asr_adapter(
                settings.model_asr,
                device=settings.asr_device,
                quantization=settings.asr_quantization,
                vad_enabled=settings.asr_onnx_vad == "enabled",
                vad_threshold=settings.asr_onnx_vad_threshold,
            )
        elif settings.backend_asr == "onnx-genai":
            from coro.backends.asr.onnx_genai import build_onnx_genai_adapter

            asr_adapter = build_onnx_genai_adapter(
                settings.model_asr,
                device=settings.asr_device,
                quantization=settings.asr_quantization,
            )
        else:
            from coro.backends.asr.faster_whisper import build_asr_adapter

            asr_adapter = build_asr_adapter(
                settings.model_asr,
                device=settings.asr_device,
                compute_type=settings.asr_compute_type,
            )
        runtime.asr_adapter = asr_adapter

        # Build optional diarization adapter
        diarization_adapter = None
        if settings.backend_diarization == "nemo" and settings.model_diarization:
            diarization_adapter = build_diarization_adapter(
                settings.model_diarization,
                device=settings.diarization_device,
            )
            runtime.diarization_adapter = diarization_adapter

            if settings.pipeline == "streaming":
                from coro.backends.diarization.nemo.streaming import NemoStreamingDiarizerFactory

                streaming_factory = NemoStreamingDiarizerFactory(
                    diarization_adapter._model,
                    tier=settings.diarization_latency,
                )
                runtime.streaming_diarizer_factory = streaming_factory
                runtime.diarization_latency = settings.diarization_latency

        # Construct the pipeline
        if settings.pipeline == "streaming":
            runtime.pipeline = StreamingPipeline(
                asr=asr_adapter,
                streaming_diarizer_factory=runtime.streaming_diarizer_factory,
                spill_dir=settings.transcript_spill_dir,
            )
        else:
            runtime.pipeline = FullMemoryPipeline(asr=asr_adapter, diarization=diarization_adapter)

        # Server Warmup
        if settings.warmup == "enabled":
            from coro.audio import AudioInput
            from coro.bench.data import WARMUP_AUDIO_PATH

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
    from coro.api.health import router as health_router
    from coro.api.v1.transcriptions import router as v1_router

    application.state.settings = settings
    application.state.runtime = runtime
    application.include_router(health_router)
    application.include_router(v1_router)

    return application


# Lightweight module-level app: default settings, no model loaded.
# Standard ASGI launch: uvicorn coro.app:app
app = create_app()
