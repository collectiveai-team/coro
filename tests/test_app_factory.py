"""Cycle 1: app factory + /health endpoint shape.

Tests that create_app() produces a FastAPI app whose /health endpoint returns
the expected shape without loading any real ASR model.
"""

import logging
import pytest
from httpx import ASGITransport, AsyncClient

from coro.app import create_app
from coro.runtime import RuntimeState
from coro.settings import ServerSettings


@pytest.fixture
def test_settings():
    """Minimal settings object suitable for create_app in tests."""
    return ServerSettings()


@pytest.fixture
def app(test_settings):
    """Create a test app with a fake RuntimeState injected (no real model)."""
    from fastapi import FastAPI

    from coro.app import create_app

    application: FastAPI = create_app(test_settings)
    # Inject RuntimeState directly — bypasses lifespan and avoids real model init.
    application.state.runtime = RuntimeState()
    return application


@pytest.mark.asyncio
async def test_health_response_has_required_keys(app):
    """GET /health returns status, ready, startup selection, and readiness keys."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "ready" in body
    assert "startup_selection" in body
    assert "capability_readiness" in body


@pytest.mark.asyncio
async def test_health_status_is_ok(app):
    """GET /health returns status='ok'."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_not_ready_when_no_asr_adapter(app):
    """GET /health returns ready=False when no ASR adapter is loaded."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.json()["ready"] is False


@pytest.mark.asyncio
async def test_warmup_disabled_skips_warmup_and_reports_ready(caplog):
    """CORO_WARMUP=disabled skips Server Warmup and reports ready."""
    from unittest.mock import patch

    from starlette.testclient import TestClient

    with patch("coro.backends.asr.faster_whisper.build_asr_adapter") as mock_build:
        mock_build.return_value = object()
        settings = ServerSettings(warmup="disabled")
        application = create_app(settings)

        with caplog.at_level(logging.WARNING, logger="coro.app"), TestClient(application) as client:
            response = client.get("/health")

    body = response.json()
    assert body["warmup_ready"] is True
    assert body["ready"] is True
    assert any("warmup" in r.message.lower() for r in caplog.records)


def test_warmup_failure_fails_server_startup():
    """Server Warmup failures fail server startup loudly."""
    from unittest.mock import AsyncMock, patch

    from starlette.testclient import TestClient

    with patch("coro.backends.asr.faster_whisper.build_asr_adapter") as mock_build:
        mock_build.return_value = object()
        settings = ServerSettings(warmup="enabled")
        application = create_app(settings)

        with (
            patch(
                "coro.pipelines.full_memory.FullMemoryPipeline.transcribe",
                new_callable=AsyncMock,
                side_effect=RuntimeError("warmup failed"),
            ),
            pytest.raises(RuntimeError, match="warmup failed"),
            TestClient(application),
        ):
            pass
