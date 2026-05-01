"""Health check endpoint for asr_diar_server."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return startup selection and capability readiness."""
    runtime = request.app.state.runtime
    return JSONResponse(
        {
            "status": "ok",
            "ready": runtime.ready,
            "startup_selection": {
                "pipeline": runtime.pipeline_selector,
                "asr_provider": runtime.asr_provider,
                "asr_model": runtime.asr_model,
                "diarization_provider": runtime.diarization_provider,
                "diarization_model": runtime.diarization_model,
            },
            "capability_readiness": {
                "asr": runtime.asr_adapter is not None,
                "diarization": (
                    "disabled"
                    if runtime.diarization_provider == "none"
                    else runtime.diarization_adapter is not None
                ),
                "transcription": runtime.ready,
            },
        }
    )
