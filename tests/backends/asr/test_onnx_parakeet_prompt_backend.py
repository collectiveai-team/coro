"""onnx-parakeet-prompt backend: prompt-kernel math, adapter, and builder wiring.

Covers:
- ``_apply_prompt_kernel``'s NumPy replication of the checkpoint's
  ``Linear -> ReLU -> Linear`` prompt-conditioning MLP, against a hand-computed
  example.
- ``_ForcedPrompt``'s set/clear of a fake ASR instance's ``_prompt_id``.
- ``OnnxParakeetPromptASRAdapter.transcribe_pcm`` against a stub ASR object
  (no real ONNX Runtime session) -- language resolution, the "no auto
  detection" error, and the serialised Adapter Concurrency Policy.
- ``build_onnx_parakeet_prompt_adapter``'s artifact-directory contract
  (missing-file errors, quantization suffix selection, prompt dictionary
  loading) with ``onnxruntime.InferenceSession`` stubbed out -- the real
  ``onnx_asr`` library and its NumPy preprocessor run unmodified; only the
  encoder/decoder_joint session construction is faked, since no real ``.onnx``
  file is available in this test environment (see the handoff notes in
  ``.scratch/issue-64-language-constrained-asr/``).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coro.backends.asr.onnx_parakeet_prompt import (
    OnnxParakeetPromptASRAdapter,
    _apply_prompt_kernel,
    _ForcedPrompt,
    build_onnx_parakeet_prompt_adapter,
)

_PROMPT_DICTIONARY = {"es-US": 0, "en-US": 1, "fr": 2}
_SAMPLE_RATE = 16000


def _pcm(seconds: float = 1.0) -> bytes:
    return b"\x00\x00" * int(_SAMPLE_RATE * seconds)


# ---------------------------------------------------------------------------
# _apply_prompt_kernel
# ---------------------------------------------------------------------------


class TestApplyPromptKernel:
    def test_matches_hand_computed_linear_relu_linear(self):
        """A tiny (hidden=2, num_prompts=2) example computed by hand."""
        encoder_out = np.array([[[1.0, 2.0]]], dtype=np.float32)  # (batch=1, time=1, hidden=2)
        weights = {
            "0.weight": np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
            "0.bias": np.array([0.0, 0.0], dtype=np.float32),
            "2.weight": np.array([[1.0, -1.0]], dtype=np.float32),
            "2.bias": np.array([0.5], dtype=np.float32),
        }
        # Worked by hand: one_hot(prompt_id=0)=[1,0] concatenated onto encoder_out
        # gives [1,2,1,0]; layer0 (Linear+ReLU) gives [2,2]; layer2 (Linear) gives 0.5.
        result = _apply_prompt_kernel(encoder_out, prompt_id=0, weights=weights)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == pytest.approx(0.5, abs=1e-6)

    def test_different_prompt_ids_change_the_output(self):
        """Forcing a different language changes the conditioned representation."""
        rng = np.random.default_rng(0)
        encoder_out = rng.standard_normal((1, 3, 4)).astype(np.float32)
        weights = {
            "0.weight": rng.standard_normal((8, 6)).astype(np.float32),
            "0.bias": rng.standard_normal(8).astype(np.float32),
            "2.weight": rng.standard_normal((4, 8)).astype(np.float32),
            "2.bias": rng.standard_normal(4).astype(np.float32),
        }
        out_a = _apply_prompt_kernel(encoder_out, prompt_id=0, weights=weights)
        out_b = _apply_prompt_kernel(encoder_out, prompt_id=1, weights=weights)
        assert not np.allclose(out_a, out_b)

    def test_same_prompt_id_is_deterministic(self):
        rng = np.random.default_rng(1)
        encoder_out = rng.standard_normal((1, 2, 4)).astype(np.float32)
        weights = {
            "0.weight": rng.standard_normal((8, 6)).astype(np.float32),
            "0.bias": rng.standard_normal(8).astype(np.float32),
            "2.weight": rng.standard_normal((4, 8)).astype(np.float32),
            "2.bias": rng.standard_normal(4).astype(np.float32),
        }
        out_a = _apply_prompt_kernel(encoder_out, prompt_id=1, weights=weights)
        out_b = _apply_prompt_kernel(encoder_out, prompt_id=1, weights=weights)
        assert not np.allclose(out_a, 0.0)  # a trivially all-zero result would also compare equal
        np.testing.assert_array_equal(out_a, out_b)

    def test_preserves_encoder_output_shape(self):
        rng = np.random.default_rng(2)
        encoder_out = rng.standard_normal((2, 5, 4)).astype(np.float32)
        weights = {
            "0.weight": rng.standard_normal((8, 7)).astype(np.float32),
            "0.bias": rng.standard_normal(8).astype(np.float32),
            "2.weight": rng.standard_normal((4, 8)).astype(np.float32),
            "2.bias": rng.standard_normal(4).astype(np.float32),
        }
        result = _apply_prompt_kernel(encoder_out, prompt_id=2, weights=weights)
        assert result.shape == encoder_out.shape


# ---------------------------------------------------------------------------
# _ForcedPrompt
# ---------------------------------------------------------------------------


class _FakeAsr:
    def __init__(self):
        self._prompt_id: int | None = None


class TestForcedPrompt:
    def test_sets_prompt_id_inside_the_context(self):
        asr = _FakeAsr()
        with _ForcedPrompt(asr, 7):
            assert asr._prompt_id == 7

    def test_clears_prompt_id_after_the_context(self):
        asr = _FakeAsr()
        with _ForcedPrompt(asr, 7):
            pass
        assert asr._prompt_id is None

    def test_clears_prompt_id_even_on_exception(self):
        asr = _FakeAsr()
        with pytest.raises(RuntimeError), _ForcedPrompt(asr, 7):
            raise RuntimeError("boom")
        assert asr._prompt_id is None


# ---------------------------------------------------------------------------
# OnnxParakeetPromptASRAdapter.transcribe_pcm
# ---------------------------------------------------------------------------


class _StubAsr:
    """Fakes the prompt-conditioned onnx_asr subclass's public surface."""

    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []
        self.forced_prompt_ids: list[int] = []

    def forced_prompt(self, prompt_id: int):
        self.forced_prompt_ids.append(prompt_id)
        return _ForcedPrompt(self, prompt_id)

    def recognize_batch(self, waveforms, waveforms_len, /, **kwargs):
        self.calls.append({"waveforms": waveforms, "waveforms_len": waveforms_len, **kwargs})
        yield self._result


def _result(tokens=(" hola", " mundo"), timestamps=(0.0, 0.5), logprobs=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        tokens=list(tokens), timestamps=list(timestamps), logprobs=logprobs, text=""
    )


class TestOnnxParakeetPromptASRAdapter:
    async def test_forces_the_resolved_prompt_id(self):
        asr = _StubAsr(_result())
        adapter = OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)
        await adapter.transcribe_pcm(_pcm(), language="es-US")
        assert asr.forced_prompt_ids == [0]

    async def test_primary_subtag_resolves_to_a_prompt_id(self):
        asr = _StubAsr(_result())
        adapter = OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)
        await adapter.transcribe_pcm(_pcm(), language="en")
        assert asr.forced_prompt_ids == [1]

    async def test_missing_language_raises(self):
        asr = _StubAsr(_result())
        adapter = OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)
        with pytest.raises(ValueError, match="auto-detection"):
            await adapter.transcribe_pcm(_pcm())

    async def test_unknown_language_raises(self):
        asr = _StubAsr(_result())
        adapter = OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)
        with pytest.raises(ValueError, match="not supported"):
            await adapter.transcribe_pcm(_pcm(), language="de")

    async def test_converts_result_to_transcript_tokens(self):
        asr = _StubAsr(_result())
        adapter = OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)
        tokens = await adapter.transcribe_pcm(_pcm(), language="es-US")
        assert [t.text for t in tokens] == [" hola", " mundo"]

    async def test_need_logprobs_is_requested(self):
        asr = _StubAsr(_result())
        adapter = OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)
        await adapter.transcribe_pcm(_pcm(), language="es-US")
        assert asr.calls == [
            {
                "waveforms": pytest.approx(np.zeros((1, _SAMPLE_RATE))),
                "waveforms_len": pytest.approx(np.array([_SAMPLE_RATE])),
                "need_logprobs": True,
            }
        ]

    def test_admission_is_serialised_to_one_permit(self):
        adapter = OnnxParakeetPromptASRAdapter(
            _StubAsr(_result()), prompt_dictionary=_PROMPT_DICTIONARY
        )
        assert adapter.admission.max_concurrency == 1

    def test_honours_prompt_is_false(self):
        assert OnnxParakeetPromptASRAdapter.honours_prompt is False


# ---------------------------------------------------------------------------
# build_onnx_parakeet_prompt_adapter
# ---------------------------------------------------------------------------


def _write_vocab(path, pieces, blank_id):
    with path.open("w", encoding="utf-8") as f:
        for idx, piece in enumerate(pieces):
            f.write(f"{piece} {idx}\n")
        f.write(f"<blk> {blank_id}\n")


def _write_prompt_kernel(directory, *, hidden=4, num_prompts=3, prompt_dictionary=None):
    rng = np.random.default_rng(0)
    weights = {
        "0.weight": rng.standard_normal((2 * hidden, hidden + num_prompts)).astype(np.float32),
        "0.bias": rng.standard_normal(2 * hidden).astype(np.float32),
        "2.weight": rng.standard_normal((hidden, 2 * hidden)).astype(np.float32),
        "2.bias": rng.standard_normal(hidden).astype(np.float32),
    }
    np.savez(directory / "prompt_kernel_cache.npz", **weights)
    metadata = {
        "vocab_size": 8270,
        "blank_id": 8270,
        "prompt_dictionary": prompt_dictionary or dict(_PROMPT_DICTIONARY),
    }
    (directory / "prompt_kernel_cache.json").write_text(json.dumps(metadata), encoding="utf-8")


def _build_artifact_dir(tmp_path):
    (tmp_path / "encoder-encoder.onnx").write_bytes(b"")
    (tmp_path / "decoder_joint-encoder.onnx").write_bytes(b"")
    _write_vocab(tmp_path / "vocab.txt", ["<unk>", "\u2581hola"], blank_id=2)
    _write_prompt_kernel(tmp_path)
    return tmp_path


class TestBuildOnnxParakeetPromptAdapter:
    def test_missing_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_onnx_parakeet_prompt_adapter(str(tmp_path))

    def test_missing_prompt_kernel_files_raise_file_not_found(self, tmp_path):
        (tmp_path / "encoder-encoder.onnx").write_bytes(b"")
        (tmp_path / "decoder_joint-encoder.onnx").write_bytes(b"")
        _write_vocab(tmp_path / "vocab.txt", ["<unk>"], blank_id=1)
        with pytest.raises(FileNotFoundError):
            build_onnx_parakeet_prompt_adapter(str(tmp_path))

    def test_builds_adapter_against_stubbed_onnxruntime(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with patch("onnxruntime.InferenceSession", autospec=True):
            adapter = build_onnx_parakeet_prompt_adapter(str(tmp_path), device="cpu")

        assert isinstance(adapter, OnnxParakeetPromptASRAdapter)
        assert adapter._prompt_dictionary == _PROMPT_DICTIONARY
        assert adapter.admission.max_concurrency == 1

    def test_quantized_encoder_filename_is_selected(self, tmp_path):
        _build_artifact_dir(tmp_path)
        (tmp_path / "encoder-encoder.static_qdq_v3.onnx").write_bytes(b"")
        session_paths: list[str] = []

        def _fake_session(path, **_kwargs):
            session_paths.append(str(path))
            return MagicMock()

        with patch("onnxruntime.InferenceSession", autospec=True, side_effect=_fake_session):
            build_onnx_parakeet_prompt_adapter(
                str(tmp_path), device="cpu", quantization="static_qdq_v3"
            )

        assert any("encoder-encoder.static_qdq_v3.onnx" in p for p in session_paths)
        assert not any(p.endswith("encoder-encoder.onnx") for p in session_paths)

    def test_missing_quantized_encoder_raises_file_not_found(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_onnx_parakeet_prompt_adapter(str(tmp_path), quantization="does-not-exist")
