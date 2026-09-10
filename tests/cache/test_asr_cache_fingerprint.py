"""Key derivation includes everything that changes a prediction, and nothing else.

Two failure modes matter and they pull in opposite directions. Omitting a
prediction-affecting input serves a result the current configuration would not
produce — a correctness bug that looks like a model regression. Including an
input that only affects rendering destroys hit rate for nothing, which is the
whole point of the feature.

Derivation is a pure function, so both are asserted directly on it.
"""

from __future__ import annotations

import pytest

from coro.cache.fingerprint import (
    CACHE_FORMAT_VERSION,
    asr_fingerprint,
    fingerprint_components,
    normalise_language,
    runtime_identity,
    window_key,
)
from coro.settings import ServerSettings

_PCM = b"\x01\x02" * 800


def _settings(**overrides) -> ServerSettings:
    """Default to a stable, non-slug backend/model pair.

    Most of this file's tests care about the fingerprint mechanism, not the
    model slug registry (ticket 06): defaulting to the raw-passthrough form
    keeps quantization fields at their own unset default (None) unless a
    test explicitly overrides one, exactly as before the default ASR Model
    Selection became a quantization-filling slug.
    """
    overrides.setdefault("backend_asr", "onnx-asr")
    overrides.setdefault("model_asr", "m")
    return ServerSettings(_env_file=None, **overrides)


def _fingerprint(**overrides) -> str:
    return asr_fingerprint(_settings(**overrides), honours_prompt=False)


# MARK: Fingerprint Inputs
def test_the_same_configuration_fingerprints_the_same():
    assert _fingerprint() == _fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_asr", "faster-whisper"),
        ("model_asr", "openai/whisper-medium"),
    ],
)
def test_a_prediction_affecting_setting_changes_the_fingerprint(field, value):
    assert _fingerprint(**{field: value}) != _fingerprint()


def test_cpu_and_cuda_fingerprint_differently():
    """Measured: the two devices agree on text but diverge on token probabilities.

    Asserted as ``cpu`` against ``cuda`` rather than against ``auto``, because
    ``auto`` resolves to one of them and would coincide with it on that host.
    """
    assert _fingerprint(asr_device="cpu") != _fingerprint(asr_device="cuda")


def test_quantization_changes_the_fingerprint_for_the_backend_that_honours_it():
    assert _fingerprint(backend_asr="onnx-asr", asr_quantization="int8") != _fingerprint(
        backend_asr="onnx-asr"
    )


def test_decoder_quantization_changes_the_fingerprint_for_the_backend_that_honours_it():
    assert _fingerprint(
        backend_asr="onnx-canary-split", asr_decoder_quantization="dynamic_v1_quint8"
    ) != _fingerprint(backend_asr="onnx-canary-split")


def test_decoder_quantization_is_a_no_op_for_a_provider_that_ignores_it():
    """asr_decoder_quantization only affects onnx-canary-split's decoder_step.onnx."""
    assert _fingerprint(
        backend_asr="onnx-asr", asr_decoder_quantization="dynamic_v1_quint8"
    ) == _fingerprint(backend_asr="onnx-asr")


def test_vad_configuration_changes_the_fingerprint():
    baseline = _fingerprint(backend_asr="onnx-asr")
    assert _fingerprint(backend_asr="onnx-asr", asr_onnx_vad="enabled") != baseline
    assert _fingerprint(backend_asr="onnx-asr", asr_onnx_vad_threshold=0.7) != baseline


def test_a_setting_the_provider_ignores_does_not_change_the_fingerprint():
    """A no-op knob must not cost a full recomputation of the file."""
    assert _fingerprint(backend_asr="onnx-asr", asr_compute_type="float16") == _fingerprint(
        backend_asr="onnx-asr"
    )


def test_concurrency_settings_do_not_change_the_fingerprint():
    """ASR was measured deterministic under concurrency, so these cannot change output."""
    baseline = _fingerprint()
    assert _fingerprint(asr_max_concurrency=4) == baseline
    assert _fingerprint(asr_max_queue_depth=8) == baseline


def test_response_and_diarization_settings_do_not_change_the_fingerprint():
    """Changing only how results are rendered must still get a full hit."""
    assert _fingerprint(backend_diarization="nemo") == _fingerprint()


def test_the_windowing_geometry_changes_the_fingerprint():
    settings = _settings()
    assert asr_fingerprint(settings, honours_prompt=False, window_seconds=20.0) != asr_fingerprint(
        settings, honours_prompt=False
    )
    assert asr_fingerprint(settings, honours_prompt=False, overlap_seconds=5.0) != asr_fingerprint(
        settings, honours_prompt=False
    )


def test_the_prompt_capability_changes_the_fingerprint():
    """A backend that starts honouring prompts must invalidate, never reuse."""
    settings = _settings()
    assert asr_fingerprint(settings, honours_prompt=True) != asr_fingerprint(
        settings, honours_prompt=False
    )


def test_the_cache_format_version_is_part_of_the_fingerprint():
    components = fingerprint_components(_settings(), honours_prompt=False)
    assert components.cache_format_version == CACHE_FORMAT_VERSION


def test_the_fingerprint_records_a_resolved_device_never_auto():
    """`auto` is not a device: it means CPU on one host and CUDA on another."""
    components = fingerprint_components(_settings(asr_device="auto"), honours_prompt=False)
    assert components.device in {"cpu", "cuda"}


def test_the_fingerprint_records_runtime_and_accelerator_identity():
    """Determinism is not portable across either, so both must be recorded."""
    components = fingerprint_components(_settings(asr_device="cpu"), honours_prompt=False)
    assert components.runtime == runtime_identity(_settings().backend_asr)
    assert components.accelerator == "cpu"


def test_nemo_runtime_identity_covers_torch_and_nemo_toolkit():
    """The PyTorch backend's determinism rides on torch and nemo-toolkit."""
    identity = runtime_identity("nemo")
    assert "torch" in identity
    assert "nemo-toolkit" in identity


# MARK: Language Normalisation
@pytest.mark.parametrize(
    ("left", "right"),
    [("es", " es "), ("es", "ES"), ("es-AR", "es_ar"), (None, ""), (None, "   ")],
)
def test_equivalent_language_spellings_normalise_together(left, right):
    assert normalise_language(left) == normalise_language(right)


def test_distinct_languages_do_not_normalise_together():
    assert normalise_language("es") != normalise_language("en")


# MARK: Window Key
def _key(**overrides) -> str:
    kwargs = {"fingerprint": "fp", "language": "es", "prompt": None, **overrides}
    pcm = kwargs.pop("pcm", _PCM)
    return window_key(pcm, **kwargs)


def test_the_same_window_under_the_same_configuration_keys_the_same():
    assert _key() == _key()


def test_different_audio_keys_differently():
    assert _key(pcm=_PCM + b"\x00\x00") != _key()


def test_a_different_language_keys_differently():
    assert _key(language="en") != _key()


def test_an_equivalent_language_spelling_keys_the_same():
    assert _key(language="ES") == _key(language="es")


def test_a_different_fingerprint_keys_differently():
    assert _key(fingerprint="other") != _key()


def test_a_prompt_keys_differently_when_it_is_supplied():
    assert _key(prompt="previous words") != _key(prompt=None)


def test_two_different_prompts_key_differently():
    assert _key(prompt="alpha") != _key(prompt="beta")
