"""Settings Dependency, Pipeline Dependency, and Singleton Runtime behavior."""

from __future__ import annotations

import pytest

from asr_diar_server.api.dependencies import get_pipeline, get_settings
from asr_diar_server.api.exceptions import TranscriptionReadinessError
from asr_diar_server.runtime import RuntimeState
from asr_diar_server.settings import ServerSettings


def test_get_settings_returns_app_settings():
    settings = ServerSettings(_env_file=None)
    request = type("Request", (), {"app": type("App", (), {"state": type("State", (), {})()})()})()
    request.app.state.settings = settings

    assert get_settings(request) is settings


def test_get_pipeline_returns_singleton_runtime_pipeline():
    pipeline = object()
    request = type("Request", (), {"app": type("App", (), {"state": type("State", (), {})()})()})()
    request.app.state.runtime = RuntimeState(pipeline=pipeline)

    assert get_pipeline(request) is pipeline


def test_get_pipeline_raises_readiness_error_when_missing():
    request = type("Request", (), {"app": type("App", (), {"state": type("State", (), {})()})()})()
    request.app.state.runtime = RuntimeState(pipeline=None)

    with pytest.raises(TranscriptionReadinessError):
        get_pipeline(request)
