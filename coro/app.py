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

from coro.api.docs import register_docs
from coro.api.errors import transcription_exception_handler
from coro.api.exceptions import TranscriptionError
from coro.runtime import RuntimeState
from coro.settings import ServerSettings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

API_TITLE = "ASR Diarization Server"

# Each contract names the other, so a consumer that only ever fetches one
# document still learns the other half exists. This is contract metadata, not UI
# copy: the pointer is just as useful to a codegen tool as to a reader.
_CONTRACT_CROSS_REFERENCE = (
    "This is the request/response half of coro's contract. The server-sent "
    "event stream returned when `stream=true` is published separately as "
    "AsyncAPI at `/asyncapi.json`; `/docs` renders both."
)


def api_metadata(summary: str, license_expression: str) -> tuple[str, dict[str, str] | None]:
    """Compose the OpenAPI ``info`` description and license from package metadata.

    A source tree with no install has no distribution metadata, so both inputs
    are empty. Each is dropped rather than published blank: an empty SPDX
    identifier is not a valid OpenAPI license object, and a description opening
    with a blank line is just noise. The CI contract gate runs in exactly that
    environment (``uv sync --only-group contracts`` installs no project), so this
    is a live path, not a defensive one.

    Args:
        summary: Distribution summary, or empty when metadata is unavailable.
        license_expression: SPDX expression, or empty when unavailable.

    Returns:
        The description, and the license object or ``None`` when unknown.

    """
    description = "\n\n".join(part for part in (summary, _CONTRACT_CROSS_REFERENCE) if part)
    license_info = (
        {"name": license_expression, "identifier": license_expression}
        if license_expression
        else None
    )
    return description, license_info


API_DESCRIPTION, API_LICENSE_INFO = api_metadata(coro.__summary__, coro.__license__)


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
        from coro.backends.asr.factory import build_asr_adapter
        from coro.backends.diarization import factory as diarization_factory
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

        # Build ASR adapter (always required) via the ASR Backend Adapter Factory.
        asr_adapter = build_asr_adapter(settings)
        runtime.asr_adapter = asr_adapter

        # Build optional diarization adapter via the diarization Backend Adapter Factory.
        diarization_adapter = None
        if settings.backend_diarization != "none":
            diarization_model = settings.model_diarization
            if not diarization_model:
                # Strict Startup Validation already rejects this combination; failing
                # here too keeps the invariant enforced at the point of use, so an
                # enabled provider can never silently degrade to an ASR-Only Server.
                msg = (
                    f"Diarization Backend Provider '{settings.backend_diarization}' is "
                    "selected but the Diarization Model Selection is empty."
                )
                raise ValueError(msg)
            diarization_adapter = diarization_factory.build_diarization_adapter(
                settings.backend_diarization,
                diarization_model,
                device=settings.diarization_device,
                hf_token=settings.hf_token.get_secret_value() if settings.hf_token else None,
                postprocessing=settings.diarization_postprocessing,
                postprocessing_max_speakers=settings.diarization_postprocessing_max_speakers,
            )
            runtime.diarization_adapter = diarization_adapter

            if settings.pipeline == "streaming" and diarization_factory.supports_streaming(
                settings.backend_diarization
            ):
                runtime.streaming_diarizer_factory = (
                    diarization_factory.build_streaming_diarizer_factory(
                        settings.backend_diarization,
                        diarization_adapter,
                        tier=settings.diarization_latency,
                    )
                )
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

    # docs_url/redoc_url are off because Scalar owns /docs: it is the only
    # renderer that can show the OpenAPI and AsyncAPI contracts together.
    application = FastAPI(
        title=API_TITLE,
        version=coro.__version__,
        description=API_DESCRIPTION,
        license_info=API_LICENSE_INFO,
        # A relative server keeps the published contract correct behind any
        # host, port or reverse proxy, and is what redocly's no-empty-servers
        # rule asks for.
        servers=[{"url": "/", "description": "This server."}],
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    application.add_exception_handler(TranscriptionError, transcription_exception_handler)
    register_docs(application, title=f"{API_TITLE} API")

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
