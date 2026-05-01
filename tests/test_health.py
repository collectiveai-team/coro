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
async def test_health_backend_matches_runtime():
    """backend in /health response matches the RuntimeState backend field."""
    app = _make_app(RuntimeState(backend="faster-whisper"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.json()["backend"] == "faster-whisper"
