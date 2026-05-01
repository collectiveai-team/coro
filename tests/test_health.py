"""Cycle 2: /health ready/not-ready behavior with injected RuntimeState.

Tests that /health accurately reflects RuntimeState.ready regardless of
how the runtime was assembled.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from asr_diar_server.app import create_app
from asr_diar_server.runtime import RuntimeState
from asr_diar_server.settings import ServerSettings


def _make_app(runtime: RuntimeState):
    """Build a test app with the given RuntimeState pre-injected."""
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    application.state.runtime = runtime
    return application


@pytest.mark.asyncio
async def test_health_ready_false_without_asr_adapter():
    """ready=False when RuntimeState has no ASR adapter."""
    app = _make_app(RuntimeState(asr_adapter=None))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is False


@pytest.mark.asyncio
async def test_health_ready_true_with_fake_asr_adapter():
    """ready=True when RuntimeState carries a non-None ASR adapter."""

    class _FakeASRAdapter:
        pass

    app = _make_app(RuntimeState(asr_adapter=_FakeASRAdapter()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True


@pytest.mark.asyncio
async def test_health_reports_startup_selection_and_capability_readiness():
    """Health separates startup selection from capability readiness."""
    runtime = RuntimeState(
        asr_adapter=object(),
        pipeline_selector="chunked-file",
        asr_provider="whisperlivekit",
        asr_model="openai/whisper-medium",
        diarization_provider="none",
        diarization_model=None,
    )
    app = _make_app(runtime)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    body = response.json()
    assert body["startup_selection"] == {
        "pipeline": "chunked-file",
        "asr_provider": "whisperlivekit",
        "asr_model": "openai/whisper-medium",
        "diarization_provider": "none",
        "diarization_model": None,
    }
    assert body["capability_readiness"] == {
        "asr": True,
        "diarization": "disabled",
        "transcription": True,
    }
