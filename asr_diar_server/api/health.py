"""Health check endpoint for asr_diar_server.

Exposes GET /health.  Returns status, ready flag, and backend identifier.
The ready flag reflects whether the ASR adapter is loaded in RuntimeState.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return server readiness and backend information.

    Returns:
        JSON with ``status``, ``ready``, and ``backend`` keys.

    """
    runtime = request.app.state.runtime
    return JSONResponse(
        {
            "status": "ok",
            "ready": runtime.ready,
            "backend": runtime.backend,
        }
    )
