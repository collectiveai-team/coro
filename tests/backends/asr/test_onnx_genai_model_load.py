"""onnx-genai model loading: what the operator sees when GenAI cannot load the model.

``onnxruntime_genai`` is replaced by a fake module so the tests exercise
``build_onnx_genai_adapter``'s own handling of a failed ``og.Model(...)`` without
the runtime or a model download.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from coro.backends.asr.onnx_genai import build_onnx_genai_adapter

# The message GenAI raises (as a RuntimeError) when an external-data file reaches
# outside the model directory through a symlink chain, as in a cache written by
# huggingface-hub >= 1 (snapshot -> blobs/<hash> -> shared blobs/<shard>/<hash>).
_ESCAPE_ERROR = (
    "External data path validation failed for initializer: enc.pre_encode.out.bias. "
    "Error: External data path escapes model directory. "
    'External data path: "encoder.onnx.data" resolved path: "/cache/blobs/ec/ec60ab"'
)


class _FakeConfig:
    def __init__(self, path: str) -> None:
        self.path = path

    def clear_providers(self) -> None:
        pass

    def append_provider(self, name: str) -> None:
        pass


def _install_fake_genai(monkeypatch, model_error: Exception) -> None:
    def _model(_config):
        raise model_error

    fake = SimpleNamespace(Config=_FakeConfig, Model=_model)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", fake)


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"chunk_samples": 8960, "sample_rate": 16000}})
    )
    return tmp_path


def test_symlinked_external_data_in_a_hub_model_is_reported_with_a_remedy(monkeypatch, model_dir):
    """A repo id resolved into a cache GenAI cannot read says how to download real files."""
    original = RuntimeError(_ESCAPE_ERROR)
    _install_fake_genai(monkeypatch, original)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda _repo_id: str(model_dir))

    with pytest.raises(RuntimeError) as excinfo:
        build_onnx_genai_adapter("org/some-model")

    message = str(excinfo.value)
    assert str(model_dir) in message
    assert "symlink" in message
    assert "hf download org/some-model --local-dir" in message
    assert excinfo.value.__cause__ is original


def test_symlinked_external_data_in_a_local_model_suggests_copying_it(monkeypatch, model_dir):
    """A local path is not a repo id, so ``hf download`` would be the wrong advice."""
    _install_fake_genai(monkeypatch, RuntimeError(_ESCAPE_ERROR))

    with pytest.raises(RuntimeError) as excinfo:
        build_onnx_genai_adapter(str(model_dir))

    message = str(excinfo.value)
    assert "symlink" in message
    assert "cp -rL" in message
    assert "hf download" not in message


def test_other_load_failures_propagate_unchanged(monkeypatch, model_dir):
    """Only the symlink-escape failure gets the remedy; anything else is not ours to explain."""
    original = RuntimeError("Failed to load model: out of memory")
    _install_fake_genai(monkeypatch, original)

    with pytest.raises(RuntimeError) as excinfo:
        build_onnx_genai_adapter(str(model_dir))

    assert excinfo.value is original
