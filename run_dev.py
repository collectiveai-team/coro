"""Development server with a fake pipeline — no ML models required.

Useful for manually testing the HTTP layer (routing, SSE framing,
error responses, response_format validation, etc.) without a GPU.

Usage:
    uv run python run_dev.py
    # or via the VS Code launch config "API: uvicorn (fake pipeline, no models)"
"""

from __future__ import annotations

import json

import uvicorn

from asr_diar_server.app import create_app
from asr_diar_server.core.types import (
    TranscriptDeltaEvent,
    TranscriptDoneEvent,
    TranscriptToken,
)
from asr_diar_server.pipelines.full_memory import FullMemoryPipeline
from asr_diar_server.settings import ServerSettings


# MARK: Fake ASR Adapter
class _FakeASR:
    """Returns a canned token list without loading any model."""

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        return [
            TranscriptToken(start=0.0, end=0.5, text=" Hello,", probability=0.95),
            TranscriptToken(start=0.5, end=1.0, text=" world.", probability=0.92),
        ]


# MARK: App Setup
app = create_app(ServerSettings(_env_file=None))


@app.on_event("startup")
async def _inject_fake_pipeline() -> None:
    app.state.runtime.pipeline = FullMemoryPipeline(asr=_FakeASR())


if __name__ == "__main__":
    uvicorn.run(
        "run_dev:app",
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        reload=False,
    )
