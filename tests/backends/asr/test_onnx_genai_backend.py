"""onnx-genai (nemotron) backend pure-logic tests.

The streaming loop requires the onnxruntime-genai runtime + model and is covered by the
benchmark harness; here we test the language mapping and language-tag stripping, plus the
reuse of the onnx-asr word reconstruction for GenAI-style decoded text pieces.
"""

from __future__ import annotations

from types import SimpleNamespace

from coro.backends.asr.onnx_asr import convert_onnx_asr_result
from coro.backends.asr.onnx_genai import _LANG_TAG_RE, _apply_device, _lang_id_for


class _RecordingConfig:
    """Stands in for ``og.Config``, recording the provider calls made on it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def clear_providers(self) -> None:
        self.calls.append(("clear",))

    def append_provider(self, name: str) -> None:
        self.calls.append(("append", name))


def _ort_providers(monkeypatch, providers):
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: providers)


def test_auto_device_selects_cuda_when_onnxruntime_offers_it(monkeypatch):
    """``auto`` must not silently leave a GPU host on CPU (~5x slower, measured)."""
    _ort_providers(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    config = _RecordingConfig()

    _apply_device(config, "auto")

    assert config.calls == [("clear",), ("append", "cuda")]


def test_auto_device_leaves_the_model_default_without_cuda(monkeypatch):
    """On a CPU-only host ``auto`` keeps following the model's own ``genai_config.json``."""
    _ort_providers(monkeypatch, ["CPUExecutionProvider"])
    config = _RecordingConfig()

    _apply_device(config, "auto")

    assert config.calls == []


def test_explicit_cpu_overrides_available_cuda(monkeypatch):
    """An explicit ``cpu`` still wins on a GPU host."""
    _ort_providers(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    config = _RecordingConfig()

    _apply_device(config, "cpu")

    assert config.calls == [("clear",)]


def test_explicit_cuda_does_not_depend_on_detection(monkeypatch):
    """An explicit ``cuda`` is honoured as before, so a misdetection can be worked around."""
    _ort_providers(monkeypatch, ["CPUExecutionProvider"])
    config = _RecordingConfig()

    _apply_device(config, "cuda")

    assert config.calls == [("clear",), ("append", "cuda")]


def test_lang_id_known_codes():
    """Known language/locale codes map to the model's lang_id."""
    assert _lang_id_for("en") == 0
    assert _lang_id_for("en-GB") == 1
    assert _lang_id_for("es") == 3
    assert _lang_id_for("auto") == 101


def test_lang_id_falls_back_to_base_then_default():
    """Unknown locale falls back to base code, then to English default."""
    assert _lang_id_for("es-AR") == 3  # base 'es'
    assert _lang_id_for("xx") == 0  # unknown -> default English
    assert _lang_id_for(None) == 0


def test_language_tag_stripping():
    """Inline language-tag tokens are removed from decoded text."""
    assert _LANG_TAG_RE.sub("", "hello <en-US> world") == "hello  world"
    assert _LANG_TAG_RE.sub("", "<es> hola") == " hola"
    assert _LANG_TAG_RE.sub("", "no tags here") == "no tags here"


def test_genai_pieces_reconstruct_words():
    """GenAI decoded pieces (space-prefixed) reconstruct into word tokens."""
    result = SimpleNamespace(
        tokens=[" hello", " world", "."],
        timestamps=[0.0, 0.56, 1.12],
        logprobs=None,
    )
    tokens = convert_onnx_asr_result(result)
    assert [t.text for t in tokens] == [" hello", " world."]
