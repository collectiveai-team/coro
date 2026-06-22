"""Server Startup Selection settings behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coro.settings import ServerSettings


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
    assert settings.asr_onnx_vad == "disabled"
    assert settings.asr_onnx_vad_threshold is None


def test_onnx_vad_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CORO_ASR_ONNX_VAD", "enabled")
    monkeypatch.setenv("CORO_ASR_ONNX_VAD_THRESHOLD", "0.4")
    settings = ServerSettings(_env_file=None)

    assert settings.asr_onnx_vad == "enabled"
    assert settings.asr_onnx_vad_threshold == 0.4


@pytest.mark.parametrize("value", ["on", "true", "yes", ""])
def test_onnx_vad_selector_is_strict(value: str):
    with pytest.raises(ValidationError):
        # Intentionally invalid value to assert strict validation.
        ServerSettings(asr_onnx_vad=value, _env_file=None)  # pyrefly: ignore[bad-argument-type]


def test_nemo_diarization_gets_default_model():
    settings = ServerSettings(
        backend_diarization="nemo",
        _env_file=None,
    )

    assert settings.model_diarization == "nvidia/diar_streaming_sortformer_4spk-v2"


def test_pyannote_diarization_gets_default_model():
    settings = ServerSettings(
        backend_diarization="pyannote",
        _env_file=None,
    )

    assert settings.model_diarization == "pyannote/speaker-diarization-community-1"


def test_pyannote_streaming_pipeline_is_rejected():
    with pytest.raises(ValidationError, match="batch-only"):
        ServerSettings(
            backend_diarization="pyannote",
            pipeline="streaming",
            _env_file=None,
        )


def test_pyannote_full_memory_pipeline_is_allowed():
    settings = ServerSettings(
        backend_diarization="pyannote",
        pipeline="full-memory",
        _env_file=None,
    )

    assert settings.backend_diarization == "pyannote"
    assert settings.pipeline == "full-memory"


@pytest.mark.parametrize("env_name", ["CORO_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_hf_token_read_from_standard_env_names(monkeypatch, env_name: str):
    monkeypatch.setenv(env_name, "secret-token")
    settings = ServerSettings(_env_file=None)

    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "secret-token"


def test_hf_token_is_masked_in_repr():
    settings = ServerSettings(hf_token="secret-token", _env_file=None)

    assert "secret-token" not in repr(settings)
    assert "secret-token" not in str(settings.model_dump())


def test_transcript_spill_dir_defaults_none_and_reads_env(monkeypatch):
    assert ServerSettings(_env_file=None).transcript_spill_dir is None
    monkeypatch.setenv("CORO_TRANSCRIPT_SPILL_DIR", "/var/lib/asr-spill")
    assert ServerSettings(_env_file=None).transcript_spill_dir == "/var/lib/asr-spill"


@pytest.mark.parametrize("pipeline", ["unknown", "v1", "v2", ""])
def test_pipeline_selector_is_strict(pipeline: str):
    with pytest.raises(ValidationError):
        # Intentionally invalid value to assert strict validation.
        ServerSettings(pipeline=pipeline, _env_file=None)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize("field", ["backend_asr", "backend_diarization"])
def test_backend_provider_selectors_are_strict(field: str):
    with pytest.raises(ValidationError):
        # Intentionally invalid value to assert strict validation.
        ServerSettings(**{field: "bogus"}, _env_file=None)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize("asr_device", ["unknown", "gpu", ""])
def test_asr_device_selector_is_strict(asr_device: str):
    with pytest.raises(ValidationError):
        # Intentionally invalid value to assert strict validation.
        ServerSettings(asr_device=asr_device, _env_file=None)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize("diarization_device", ["unknown", "gpu", ""])
def test_diarization_device_selector_is_strict(diarization_device: str):
    with pytest.raises(ValidationError):
        # Intentionally invalid value to assert strict validation.
        ServerSettings(
            diarization_device=diarization_device,  # pyrefly: ignore[bad-argument-type]
            _env_file=None,
        )
