"""Cycle 2: /health ready/not-ready behavior with injected RuntimeState.

Tests that /health accurately reflects RuntimeState.ready regardless of
how the runtime was assembled.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

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
    """ready=True when RuntimeState carries a non-None ASR adapter and warmup is complete."""

    class _FakeASRAdapter:
        pass

    app = _make_app(RuntimeState(asr_adapter=_FakeASRAdapter(), warmup_ready=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_runtime_state_warmup_ready_defaults_false():
    """warmup_ready defaults to False on RuntimeState."""
    rt = RuntimeState()
    assert rt.warmup_ready is False


def test_server_settings_warmup_default_enabled():
    """ServerSettings.warmup defaults to 'enabled'."""
    s = ServerSettings()
    assert s.warmup == "enabled"


def test_server_settings_warmup_accepts_disabled():
    """ServerSettings.warmup accepts 'disabled'."""
    s = ServerSettings(warmup="disabled")
    assert s.warmup == "disabled"


def test_server_settings_warmup_invalid_raises():
    """Strict Startup Validation rejects unknown warmup values."""
    with pytest.raises(ValidationError):
        ServerSettings(warmup="invalid")


@pytest.mark.asyncio
async def test_health_includes_warmup_ready_key():
    """GET /health includes warmup_ready reflecting RuntimeState."""
    app = _make_app(RuntimeState(asr_adapter=object()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    body = response.json()
    assert "warmup_ready" in body
    assert body["warmup_ready"] is False


@pytest.mark.asyncio
async def test_health_reports_startup_selection_and_capability_readiness():
    """Health separates startup selection from capability readiness."""
    runtime = RuntimeState(
        asr_adapter=object(),
        pipeline_selector="chunked-file",
        asr_provider="faster-whisper",
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
        "asr_provider": "faster-whisper",
        "asr_model": "openai/whisper-medium",
        "diarization_provider": "none",
        "diarization_model": None,
    }
    assert body["capability_readiness"] == {
        "asr": True,
        "diarization": "disabled",
        "transcription": True,
    }
