"""Server Startup Selection settings behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coro.settings import DEFAULT_MODEL_SLUG, MODEL_SLUGS, ServerSettings


def test_settings_default_to_full_memory_canary_configuration():
    """Canary-1b-v2 (INT8/INT8) is the default ASR Model Selection (ADR 0019/0020)."""
    settings = ServerSettings(_env_file=None)

    assert settings.pipeline == "full-memory"
    assert settings.backend_asr == "onnx-canary-split"
    assert settings.model_asr == "collectiveai/canary-1b-v2-onnx-split-int8"
    assert settings.asr_quantization == "static_qdq_v4_pct_excl"
    assert settings.asr_decoder_quantization == "dynamic_v1_quint8"
    assert settings.asr_device == "auto"
    assert settings.asr_compute_type == "default"
    assert settings.backend_diarization == "none"
    assert settings.model_diarization is None
    assert settings.diarization_device == "auto"
    assert settings.asr_onnx_vad == "disabled"
    assert settings.asr_onnx_vad_threshold is None
    assert settings.diarization_postprocessing is None
    assert settings.diarization_postprocessing_max_speakers == 4


def test_a_non_slug_asr_quantization_stays_unset_for_a_non_canary_backend():
    """int8 is a memory-fitting tool, not a speed tool, for the parakeet slug."""
    settings = ServerSettings(_env_file=None, model_asr="parakeet-tdt-0.6b-v3")
    assert settings.asr_quantization is None
    assert settings.asr_decoder_quantization is None


# ---------------------------------------------------------------------------
# Model Slug Registry (ADR 0020)
# ---------------------------------------------------------------------------


class TestModelSlugRegistry:
    def test_parakeet_slug_resolves_backend_and_model_with_no_quantization(self):
        settings = ServerSettings(_env_file=None, model_asr="parakeet-tdt-0.6b-v3")
        assert settings.backend_asr == "onnx-asr"
        assert settings.model_asr == "nemo-parakeet-tdt-0.6b-v3"
        assert settings.asr_quantization is None
        assert settings.asr_decoder_quantization is None

    def test_whisper_turbo_slug_resolves_backend_and_model(self):
        settings = ServerSettings(_env_file=None, model_asr="whisper-large-v3-turbo")
        assert settings.backend_asr == "faster-whisper"
        assert settings.model_asr == "large-v3-turbo"

    def test_whisper_slug_resolves_backend_and_model(self):
        settings = ServerSettings(_env_file=None, model_asr="whisper-large-v3")
        assert settings.backend_asr == "faster-whisper"
        assert settings.model_asr == "large-v3"

    def test_explicit_backend_and_raw_model_id_is_unchanged_from_before_slugs(self):
        """`--backend-asr onnx-asr --model-asr nemo-parakeet-tdt-0.6b-v3` is a no-op change."""
        # Oracle computed via a different code path (the slug), not duplicated
        # from this test's own constructor call.
        via_slug = ServerSettings(_env_file=None, model_asr="parakeet-tdt-0.6b-v3")
        settings = ServerSettings(
            _env_file=None, backend_asr="onnx-asr", model_asr="nemo-parakeet-tdt-0.6b-v3"
        )
        assert settings.backend_asr == via_slug.backend_asr
        assert settings.model_asr == via_slug.model_asr
        assert settings.asr_quantization is None
        assert settings.asr_decoder_quantization is None

    def test_unknown_model_id_without_a_backend_raises_mentioning_backend_asr(self):
        with pytest.raises(ValidationError, match="backend_asr"):
            ServerSettings(_env_file=None, model_asr="some/unknown-id")

    def test_unknown_model_id_with_an_explicit_backend_passes_through_verbatim(self):
        raw_backend, raw_model = "onnx-genai", "some/unknown-id"
        assert raw_model not in MODEL_SLUGS  # sanity: genuinely unknown, not a slug in disguise
        settings = ServerSettings(_env_file=None, backend_asr=raw_backend, model_asr=raw_model)
        assert settings.backend_asr == raw_backend
        assert settings.model_asr == raw_model
        assert settings.asr_quantization is None
        assert settings.asr_decoder_quantization is None

    def test_fp32_overrides_the_default_slugs_encoder_quantization_but_keeps_the_decoder(self):
        settings = ServerSettings(_env_file=None, asr_quantization="fp32")
        assert settings.backend_asr == "onnx-canary-split"
        assert settings.asr_quantization is None
        assert settings.asr_decoder_quantization == "dynamic_v1_quint8"

    def test_fp32_overrides_the_default_slugs_decoder_quantization_but_keeps_the_encoder(self):
        settings = ServerSettings(_env_file=None, asr_decoder_quantization="fp32")
        assert settings.asr_decoder_quantization is None
        assert settings.asr_quantization == "static_qdq_v4_pct_excl"

    def test_explicit_backend_overriding_the_default_slug_still_fills_its_quantization(self):
        """Precedence is per field: an overridden backend_asr does not un-fill model_asr's slug."""
        default_slug = MODEL_SLUGS[DEFAULT_MODEL_SLUG]
        settings = ServerSettings(_env_file=None, backend_asr="onnx-asr")
        # The oracle is the default slug's own backend, not this test's input literal:
        # proves the slug did NOT clobber the explicit backend_asr back to its own default.
        assert settings.backend_asr != default_slug["backend_asr"]
        assert settings.model_asr == default_slug["model_asr"]
        assert settings.asr_quantization == default_slug["asr_quantization"]

    def test_explicit_quantization_survives_the_default_slug(self):
        default_slug = MODEL_SLUGS[DEFAULT_MODEL_SLUG]
        settings = ServerSettings(_env_file=None, asr_quantization="int8")
        # Oracle is the slug's own quantization default, proving the explicit
        # value survived rather than being overwritten by the slug.
        assert settings.asr_quantization != default_slug["asr_quantization"]
        assert settings.asr_decoder_quantization == default_slug["asr_decoder_quantization"]


def test_onnx_vad_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CORO_ASR_ONNX_VAD", "enabled")
    monkeypatch.setenv("CORO_ASR_ONNX_VAD_THRESHOLD", "0.4")
    settings = ServerSettings(_env_file=None)

    assert settings.asr_onnx_vad == "enabled"
    assert settings.asr_onnx_vad_threshold == 0.4


def test_asr_fallback_language_defaults_to_english():
    assert ServerSettings(_env_file=None).asr_fallback_language == "en"


def test_asr_fallback_language_env_flip(monkeypatch):
    monkeypatch.setenv("CORO_ASR_FALLBACK_LANGUAGE", "es")
    assert ServerSettings(_env_file=None).asr_fallback_language == "es"


def test_asr_fallback_language_blank_collapses_to_the_default(monkeypatch):
    monkeypatch.setenv("CORO_ASR_FALLBACK_LANGUAGE", "   ")
    assert ServerSettings(_env_file=None).asr_fallback_language == "en"


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


@pytest.mark.parametrize("backend", ["nemo", "pyannote"])
@pytest.mark.parametrize("model", ["", "   "])
def test_enabled_diarization_with_empty_model_is_rejected(backend: str, model: str):
    """An empty Diarization Model Selection must fail loudly, not degrade to ASR-only."""
    with pytest.raises(ValidationError, match="Diarization Model Selection is empty"):
        ServerSettings(
            backend_diarization=backend,  # pyrefly: ignore[bad-argument-type]
            model_diarization=model,
            _env_file=None,
        )


def test_enabled_diarization_with_empty_model_from_env_is_rejected(monkeypatch):
    monkeypatch.setenv("CORO_BACKEND_DIARIZATION", "nemo")
    monkeypatch.setenv("CORO_MODEL_DIARIZATION", "")

    with pytest.raises(ValidationError, match="Diarization Model Selection is empty"):
        ServerSettings(_env_file=None)


def test_disabled_diarization_with_empty_model_is_allowed():
    """An ASR-Only Server is a valid configuration and stays valid."""
    settings = ServerSettings(
        backend_diarization="none",
        model_diarization="",
        _env_file=None,
    )

    assert settings.backend_diarization == "none"


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


def test_diarization_postprocessing_defaults_none_and_reads_env(monkeypatch):
    """Unrestricted like model_diarization: a preset name or a custom path, resolved
    by the NeMo adapter (see ADR 0010) — settings itself does not validate the value."""
    assert ServerSettings(_env_file=None).diarization_postprocessing is None
    monkeypatch.setenv("CORO_DIARIZATION_POSTPROCESSING", "dihard3-dev")
    assert ServerSettings(_env_file=None).diarization_postprocessing == "dihard3-dev"


@pytest.mark.parametrize("opt_out", ["none", ""])
def test_diarization_postprocessing_can_be_returned_to_the_nemo_baseline(monkeypatch, opt_out):
    """Spelling the baseline explicitly must resolve, not fail startup.

    The default is already the baseline, but an operator templating the variable
    can only unset it by writing something; '' and 'none' are the two spellings
    that must not raise.
    """
    from coro.backends.diarization.nemo.postprocessing import resolve_postprocessing_yaml

    monkeypatch.setenv("CORO_DIARIZATION_POSTPROCESSING", opt_out)
    value = ServerSettings(_env_file=None).diarization_postprocessing
    assert resolve_postprocessing_yaml(value) is None


def test_diarization_postprocessing_max_speakers_reads_env(monkeypatch):
    monkeypatch.setenv("CORO_DIARIZATION_POSTPROCESSING_MAX_SPEAKERS", "8")
    assert ServerSettings(_env_file=None).diarization_postprocessing_max_speakers == 8


def test_diarization_postprocessing_max_speakers_rejects_zero():
    with pytest.raises(ValidationError):
        ServerSettings(diarization_postprocessing_max_speakers=0, _env_file=None)


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
