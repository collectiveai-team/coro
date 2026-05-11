"""Server Startup Selection settings behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from asr_diar_server.settings import ServerSettings


def test_settings_default_to_full_memory_asr_only_configuration():
    settings = ServerSettings(_env_file=None)

    assert settings.pipeline == "full-memory"
    assert settings.backend_asr == "faster-whisper"
    assert settings.model_asr == "openai/whisper-medium"
    assert settings.asr_device == "auto"
    assert settings.asr_compute_type == "default"
    assert settings.backend_diarization == "none"
    assert settings.model_diarization is None
    assert settings.diarization_device == "auto"


def test_nemo_diarization_gets_default_model():
    settings = ServerSettings(
        backend_diarization="nemo",
        _env_file=None,
    )

    assert settings.model_diarization == "nvidia/diar_streaming_sortformer_4spk-v2"


@pytest.mark.parametrize("pipeline", ["unknown", "v1", "v2", ""])
def test_pipeline_selector_is_strict(pipeline: str):
    with pytest.raises(ValidationError):
        ServerSettings(pipeline=pipeline, _env_file=None)


@pytest.mark.parametrize("field", ["backend_asr", "backend_diarization"])
def test_backend_provider_selectors_are_strict(field: str):
    with pytest.raises(ValidationError):
        ServerSettings(**{field: "bogus"}, _env_file=None)


@pytest.mark.parametrize("asr_device", ["unknown", "gpu", ""])
def test_asr_device_selector_is_strict(asr_device: str):
    with pytest.raises(ValidationError):
        ServerSettings(asr_device=asr_device, _env_file=None)


@pytest.mark.parametrize("diarization_device", ["unknown", "gpu", ""])
def test_diarization_device_selector_is_strict(diarization_device: str):
    with pytest.raises(ValidationError):
        ServerSettings(diarization_device=diarization_device, _env_file=None)
