"""The app factory: /health shape, Server Warmup and diarization at startup.

Startup goes through the ASR and Diarization Backend Adapter Factories, which
serve in-memory adapters here, so no model is loaded and no network is used.
"""

from __future__ import annotations

import logging

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from coro.app import create_app
from coro.backends.asr import factory as asr_factory
from coro.backends.diarization import factory as diarization_factory
from coro.core.models import SpeakerSegment
from coro.settings import ServerSettings
from support.adapters import FixedDiarization, RecordingASR

CANARY = {
    "backend_asr": "onnx-canary-split",
    "model_asr": "collectiveai/canary-1b-v2-onnx-split-int8",
    "asr_fallback_language": "es",
}
WHISPER = {"backend_asr": "faster-whisper", "model_asr": "tiny", "asr_fallback_language": "es"}


@pytest.fixture
def served_asr(monkeypatch) -> RecordingASR:
    """Serve the ASR Backend Adapter Factory an in-memory adapter."""
    adapter = RecordingASR()
    monkeypatch.setattr(asr_factory, "build_asr_adapter", lambda settings: adapter)
    return adapter


def start(settings: ServerSettings):
    """Start the app through its lifespan and return the /health body."""
    with TestClient(create_app(settings)) as client:
        return client.get("/health").json()


@pytest.fixture
async def unstarted_health():
    """/health of an app whose lifespan has not run, so no ASR Adapter is loaded."""
    transport = ASGITransport(app=create_app(ServerSettings()))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_health_response_has_required_keys(unstarted_health):
    assert {"status", "ready", "startup_selection", "capability_readiness"} <= set(unstarted_health)


@pytest.mark.asyncio
async def test_health_status_is_ok(unstarted_health):
    assert unstarted_health["status"] == "ok"


@pytest.mark.asyncio
async def test_health_not_ready_when_no_asr_adapter(unstarted_health):
    assert unstarted_health["ready"] is False


def test_warmup_disabled_reports_ready_without_transcribing(served_asr):
    health = start(ServerSettings(_env_file=None, warmup="disabled"))

    assert (health["warmup_ready"], health["ready"]) == (True, True)
    assert served_asr.window_sizes == []


def test_warmup_disabled_logs_that_warmup_was_skipped(served_asr, caplog):
    with caplog.at_level(logging.WARNING, logger="coro.app"):
        start(ServerSettings(_env_file=None, warmup="disabled"))

    assert any("warmup" in r.message.lower() for r in caplog.records)


def test_warmup_transcribes_canary_in_the_fallback_language(served_asr):
    health = start(ServerSettings(_env_file=None, warmup="enabled", **CANARY))

    assert health["warmup_ready"] is True
    assert set(served_asr.languages) == {"es"}


def test_warmup_logs_the_fallback_language_for_canary(served_asr, caplog):
    with caplog.at_level(logging.INFO, logger="coro.app"):
        start(ServerSettings(_env_file=None, warmup="enabled", **CANARY))

    assert any(
        "warmup" in r.message.lower() and "language=es" in r.message.lower() for r in caplog.records
    )


def test_warmup_omits_the_language_hint_for_a_non_canary_backend(served_asr):
    start(ServerSettings(_env_file=None, warmup="enabled", **WHISPER))

    assert set(served_asr.languages) == {None}


def test_enabled_diarization_builds_a_diarization_adapter(served_asr, monkeypatch):
    """An enabled Backend Provider always builds an adapter, never silently ASR-only."""
    diarization = FixedDiarization([SpeakerSegment(start=0.0, end=1.0, speaker=0)])
    monkeypatch.setattr(
        diarization_factory, "build_diarization_adapter", lambda *args, **kwargs: diarization
    )

    health = start(ServerSettings(_env_file=None, backend_diarization="nemo", **WHISPER))

    assert health["capability_readiness"]["diarization"] is True
    assert len(diarization.diarized_bytes) == 1


def test_warmup_failure_fails_server_startup(monkeypatch):
    failing = RecordingASR(error=RuntimeError("warmup failed"))
    monkeypatch.setattr(asr_factory, "build_asr_adapter", lambda settings: failing)

    with pytest.raises(RuntimeError, match="warmup failed"):
        start(ServerSettings(_env_file=None, warmup="enabled", **WHISPER))
