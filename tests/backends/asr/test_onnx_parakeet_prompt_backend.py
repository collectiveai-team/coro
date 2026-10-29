"""onnx-parakeet-prompt backend: prompt-kernel math, adapter, and builder wiring.

Covers:
- ``apply_prompt_kernel``'s NumPy replication of the checkpoint's
  ``Linear -> ReLU -> Linear`` prompt-conditioning MLP, against a hand-computed
  example.
- ``ForcedPrompt``'s set/clear of a fake ASR instance's ``_prompt_id``.
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
import math
from types import SimpleNamespace

import numpy as np
import pytest

from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.onnx_parakeet_prompt import (
    OnnxParakeetPromptASRAdapter,
    build_onnx_parakeet_prompt_adapter,
)
from coro.backends.asr.onnx_parakeet_prompt_kernel import ForcedPrompt, apply_prompt_kernel

_PROMPT_DICTIONARY = {"es-US": 0, "en-US": 1, "fr": 2}
_SAMPLE_RATE = 16000


def _pcm(seconds: float = 1.0) -> bytes:
    return b"\x00\x00" * int(_SAMPLE_RATE * seconds)


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
        result = apply_prompt_kernel(encoder_out, prompt_id=0, weights=weights)
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
        out_a = apply_prompt_kernel(encoder_out, prompt_id=0, weights=weights)
        out_b = apply_prompt_kernel(encoder_out, prompt_id=1, weights=weights)
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
        out_a = apply_prompt_kernel(encoder_out, prompt_id=1, weights=weights)
        out_b = apply_prompt_kernel(encoder_out, prompt_id=1, weights=weights)
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
        result = apply_prompt_kernel(encoder_out, prompt_id=2, weights=weights)
        assert result.shape == encoder_out.shape


class _PromptConditionedAsr:
    """Fakes the prompt-conditioned onnx_asr subclass: ``forced_prompt`` and ``recognize_batch``.

    Like the real subclass's ``_encode``, recognition reads the call-scoped
    ``_prompt_id`` slot that ``ForcedPrompt`` sets, and records each prompt id
    it ran under. It returns per-token log-probabilities only when asked.
    """

    def __init__(self, tokens=(" hola", " mundo"), *, fails=False):
        self._prompt_id: int | None = None
        self._tokens = list(tokens)
        self._fails = fails
        self.prompts_heard: list[int | None] = []
        self.samples_heard: list[int] = []

    def forced_prompt(self, prompt_id: int) -> ForcedPrompt:
        return ForcedPrompt(self, prompt_id)

    def prompt_in_effect(self) -> int | None:
        return self._prompt_id

    def recognize_batch(self, waveforms, waveforms_len, /, *, need_logprobs=False):
        self.prompts_heard.append(self._prompt_id)
        self.samples_heard.append(int(waveforms_len[0]))
        if self._fails:
            raise RuntimeError("boom")
        logprobs = [-0.1] * len(self._tokens) if need_logprobs else None
        timestamps = [0.5 * i for i in range(len(self._tokens))]
        yield SimpleNamespace(
            tokens=self._tokens, timestamps=timestamps, logprobs=logprobs, text=""
        )


class TestForcedPrompt:
    def test_sets_prompt_id_inside_the_context(self):
        asr = _PromptConditionedAsr()
        with ForcedPrompt(asr, 7):
            assert asr.prompt_in_effect() == 7

    def test_clears_prompt_id_after_the_context(self):
        asr = _PromptConditionedAsr()
        with ForcedPrompt(asr, 7):
            pass
        assert asr.prompt_in_effect() is None

    def test_clears_prompt_id_even_on_exception(self):
        asr = _PromptConditionedAsr()
        with pytest.raises(RuntimeError), ForcedPrompt(asr, 7):
            raise RuntimeError("boom")
        assert asr.prompt_in_effect() is None


def _adapter(asr) -> OnnxParakeetPromptASRAdapter:
    return OnnxParakeetPromptASRAdapter(asr, prompt_dictionary=_PROMPT_DICTIONARY)


class TestOnnxParakeetPromptASRAdapter:
    async def test_recognises_under_the_resolved_prompt_id(self):
        asr = _PromptConditionedAsr()
        await _adapter(asr).transcribe_pcm(_pcm(), language="es-US")
        assert asr.prompts_heard == [0]

    async def test_primary_subtag_resolves_to_a_prompt_id(self):
        asr = _PromptConditionedAsr()
        await _adapter(asr).transcribe_pcm(_pcm(), language="en")
        assert asr.prompts_heard == [1]

    async def test_the_forced_prompt_ends_with_the_call(self):
        asr = _PromptConditionedAsr()
        await _adapter(asr).transcribe_pcm(_pcm(), language="fr")
        assert asr.prompt_in_effect() is None

    async def test_the_forced_prompt_ends_when_recognition_fails(self):
        asr = _PromptConditionedAsr(fails=True)
        with pytest.raises(RuntimeError, match="boom"):
            await _adapter(asr).transcribe_pcm(_pcm(), language="fr")
        assert asr.prompt_in_effect() is None

    async def test_missing_language_raises(self):
        with pytest.raises(AsrUnsupportedLanguageError) as excinfo:
            await _adapter(_PromptConditionedAsr()).transcribe_pcm(_pcm())
        assert excinfo.value.language == ""
        assert set(excinfo.value.supported_languages) == set(_PROMPT_DICTIONARY)

    async def test_unknown_language_raises(self):
        with pytest.raises(AsrUnsupportedLanguageError, match="not supported"):
            await _adapter(_PromptConditionedAsr()).transcribe_pcm(_pcm(), language="de")

    async def test_converts_result_to_transcript_tokens(self):
        tokens = await _adapter(_PromptConditionedAsr()).transcribe_pcm(_pcm(), language="es-US")
        assert [t.text for t in tokens] == [" hola", " mundo"]

    async def test_tokens_carry_the_probability_of_their_logprobs(self):
        tokens = await _adapter(_PromptConditionedAsr()).transcribe_pcm(_pcm(), language="es-US")
        assert [t.probability for t in tokens] == pytest.approx([math.exp(-0.1)] * 2, abs=1e-6)

    async def test_the_whole_clip_is_recognised_as_one_waveform(self):
        asr = _PromptConditionedAsr()
        await _adapter(asr).transcribe_pcm(_pcm(), language="es-US")
        assert asr.samples_heard == [_SAMPLE_RATE]

    def test_admission_is_serialised_to_one_permit(self):
        assert _adapter(_PromptConditionedAsr()).admission.max_concurrency == 1

    def test_honours_prompt_is_false(self):
        assert OnnxParakeetPromptASRAdapter.honours_prompt is False


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


@pytest.fixture()
def opened_sessions(monkeypatch):
    """Paths of the ONNX Runtime sessions the builder opens; the files are empty, so none runs."""
    paths: list[str] = []

    class _InferenceSession:
        def __init__(self, path, **_kwargs):
            paths.append(str(path))

    monkeypatch.setattr("onnxruntime.InferenceSession", _InferenceSession)
    return paths


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

    async def test_builds_a_serialised_adapter_for_the_artifact_languages(
        self, tmp_path, opened_sessions
    ):
        adapter = build_onnx_parakeet_prompt_adapter(
            str(_build_artifact_dir(tmp_path)), device="cpu"
        )

        with pytest.raises(AsrUnsupportedLanguageError) as excinfo:
            await adapter.transcribe_pcm(_pcm())

        assert isinstance(adapter, OnnxParakeetPromptASRAdapter)
        assert set(excinfo.value.supported_languages) == set(_PROMPT_DICTIONARY)
        assert adapter.admission.max_concurrency == 1

    def test_quantized_encoder_filename_is_selected(self, tmp_path, opened_sessions):
        _build_artifact_dir(tmp_path)
        (tmp_path / "encoder-encoder.static_qdq_v3.onnx").write_bytes(b"")

        build_onnx_parakeet_prompt_adapter(
            str(tmp_path), device="cpu", quantization="static_qdq_v3"
        )

        assert any("encoder-encoder.static_qdq_v3.onnx" in p for p in opened_sessions)
        assert not any(p.endswith("encoder-encoder.onnx") for p in opened_sessions)

    def test_missing_quantized_encoder_raises_file_not_found(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_onnx_parakeet_prompt_adapter(str(tmp_path), quantization="does-not-exist")
