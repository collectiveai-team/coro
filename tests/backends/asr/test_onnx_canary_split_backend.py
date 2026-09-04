"""onnx-canary-split backend: adapter and builder wiring.

Covers:
- ``OnnxCanarySplitASRAdapter.transcribe_pcm`` against a stub ASR object (no
  real ONNX Runtime session) -- language forwarding, the serialised Adapter
  Concurrency Policy, and ``prompt`` being accepted but ignored (no carried-
  prompt input port on this AED model).
- ``build_onnx_canary_split_adapter``'s artifact-directory contract (missing-
  file errors, quantization suffix selection) with
  ``onnxruntime.InferenceSession`` stubbed out -- the real ``onnx_asr``
  library runs unmodified; only session construction is faked, since no real
  ``.onnx`` file is available in this test environment (same convention as
  ``test_onnx_parakeet_prompt_backend.py``, this backend's closest sibling).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coro.backends.asr.onnx_canary_split import (
    OnnxCanarySplitASRAdapter,
    build_onnx_canary_split_adapter,
)

_SAMPLE_RATE = 16000


def _pcm(seconds: float = 1.0) -> bytes:
    return b"\x00\x00" * int(_SAMPLE_RATE * seconds)


def _result(tokens=(" hola", " mundo"), timestamps=None, logprobs=None):
    from types import SimpleNamespace

    return SimpleNamespace(tokens=list(tokens), timestamps=timestamps, logprobs=logprobs, text="")


# ---------------------------------------------------------------------------
# OnnxCanarySplitASRAdapter.transcribe_pcm
# ---------------------------------------------------------------------------


class _StubAsr:
    """Fakes the split-decode ``NemoConformerAED`` subclass's public surface."""

    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def recognize_batch(self, waveforms, waveforms_len, /, **kwargs):
        self.calls.append({"waveforms": waveforms, "waveforms_len": waveforms_len, **kwargs})
        yield self._result


class TestOnnxCanarySplitASRAdapter:
    async def test_forwards_the_language_kwarg(self):
        asr = _StubAsr(_result())
        adapter = OnnxCanarySplitASRAdapter(asr)
        await adapter.transcribe_pcm(_pcm(), language="es")
        assert asr.calls == [
            {
                "waveforms": pytest.approx(np.zeros((1, _SAMPLE_RATE))),
                "waveforms_len": pytest.approx(np.array([_SAMPLE_RATE])),
                "language": "es",
            }
        ]

    async def test_no_language_omits_the_kwarg(self):
        """Canary's `_decoding` defaults its own prefix tokens when `language` is absent."""
        asr = _StubAsr(_result())
        adapter = OnnxCanarySplitASRAdapter(asr)
        await adapter.transcribe_pcm(_pcm())
        assert "language" not in asr.calls[0]

    async def test_prompt_is_accepted_but_ignored(self):
        """No carried-prompt input port on this AED model (only language/pnc prefix tokens)."""
        asr = _StubAsr(_result())
        adapter = OnnxCanarySplitASRAdapter(asr)
        await adapter.transcribe_pcm(_pcm(), language="es", prompt="some carried prompt")
        assert "prompt" not in asr.calls[0]

    async def test_converts_result_to_transcript_tokens_via_words_from_text_fallback(self):
        """No per-token timestamps for this AED model -- falls back like onnx-asr's Canary."""
        asr = _StubAsr(_result(tokens=[" hola", " mundo"], timestamps=None))
        adapter = OnnxCanarySplitASRAdapter(asr)
        tokens = await adapter.transcribe_pcm(_pcm(2.0), language="es")
        assert (
            "".join(t.text for t in tokens).strip() == ""
        )  # text-only result has no `.text` set on the stub

    def test_admission_is_serialised_to_one_permit(self):
        adapter = OnnxCanarySplitASRAdapter(_StubAsr(_result()))
        assert adapter.admission.max_concurrency == 1

    def test_honours_prompt_is_false(self):
        assert OnnxCanarySplitASRAdapter.honours_prompt is False


# ---------------------------------------------------------------------------
# build_onnx_canary_split_adapter
# ---------------------------------------------------------------------------


def _write_vocab(path, pieces, blank_id):
    with path.open("w", encoding="utf-8") as f:
        for idx, piece in enumerate(pieces):
            f.write(f"{piece} {idx}\n")
        f.write(f"<blk> {blank_id}\n")


_REQUIRED_TOKENS = [
    "<unk>",
    "\u2581",  # SentencePiece space marker; onnx_asr's vocab loader maps it to " "
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
    "<|en|>",
    "<|pnc|>",
    "<|noitn|>",
    "<|notimestamp|>",
    "<|nodiarize|>",
    "<|endoftext|>",
]


def _build_artifact_dir(tmp_path):
    (tmp_path / "encoder-model.onnx").write_bytes(b"")
    (tmp_path / "xattn_kv.onnx").write_bytes(b"")
    (tmp_path / "decoder_step.onnx").write_bytes(b"")
    _write_vocab(tmp_path / "vocab.txt", _REQUIRED_TOKENS, blank_id=len(_REQUIRED_TOKENS))
    return tmp_path


class TestBuildOnnxCanarySplitAdapter:
    def test_missing_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_onnx_canary_split_adapter(str(tmp_path))

    def test_missing_vocab_raises_file_not_found(self, tmp_path):
        (tmp_path / "encoder-model.onnx").write_bytes(b"")
        (tmp_path / "xattn_kv.onnx").write_bytes(b"")
        (tmp_path / "decoder_step.onnx").write_bytes(b"")
        with pytest.raises(FileNotFoundError):
            build_onnx_canary_split_adapter(str(tmp_path))

    def test_builds_adapter_against_stubbed_onnxruntime(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with patch("onnxruntime.InferenceSession", autospec=True, return_value=MagicMock()):
            adapter = build_onnx_canary_split_adapter(str(tmp_path), device="cpu")

        assert isinstance(adapter, OnnxCanarySplitASRAdapter)
        assert adapter.admission.max_concurrency == 1

    def test_quantized_encoder_filename_is_selected(self, tmp_path):
        _build_artifact_dir(tmp_path)
        (tmp_path / "encoder-model.static_qdq_v3.onnx").write_bytes(b"")
        session_paths: list[str] = []

        def _fake_session(path, **_kwargs):
            session_paths.append(str(path))
            return MagicMock()

        with patch("onnxruntime.InferenceSession", autospec=True, side_effect=_fake_session):
            build_onnx_canary_split_adapter(
                str(tmp_path), device="cpu", quantization="static_qdq_v3"
            )

        assert any("encoder-model.static_qdq_v3.onnx" in p for p in session_paths)
        assert not any(p.endswith("encoder-model.onnx") for p in session_paths)

    def test_missing_quantized_encoder_raises_file_not_found(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_onnx_canary_split_adapter(str(tmp_path), quantization="does-not-exist")

    def test_quantized_decoder_step_filename_is_selected(self, tmp_path):
        _build_artifact_dir(tmp_path)
        (tmp_path / "decoder_step.dynamic_v1_quint8.onnx").write_bytes(b"")
        session_paths: list[str] = []

        def _fake_session(path, **_kwargs):
            session_paths.append(str(path))
            return MagicMock()

        with patch("onnxruntime.InferenceSession", autospec=True, side_effect=_fake_session):
            build_onnx_canary_split_adapter(
                str(tmp_path), device="cpu", decoder_quantization="dynamic_v1_quint8"
            )

        assert any("decoder_step.dynamic_v1_quint8.onnx" in p for p in session_paths)
        assert not any(p.endswith("decoder_step.onnx") for p in session_paths)

    def test_missing_quantized_decoder_step_raises_file_not_found(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_onnx_canary_split_adapter(str(tmp_path), decoder_quantization="does-not-exist")

    def test_encoder_and_decoder_quantization_selectors_are_independent(self, tmp_path):
        """The two selectors pick their own artifacts without interfering."""
        _build_artifact_dir(tmp_path)
        (tmp_path / "encoder-model.static_qdq_v3.onnx").write_bytes(b"")
        (tmp_path / "decoder_step.dynamic_v1_quint8.onnx").write_bytes(b"")
        session_paths: list[str] = []

        def _fake_session(path, **_kwargs):
            session_paths.append(str(path))
            return MagicMock()

        with patch("onnxruntime.InferenceSession", autospec=True, side_effect=_fake_session):
            build_onnx_canary_split_adapter(
                str(tmp_path),
                device="cpu",
                quantization="static_qdq_v3",
                decoder_quantization="dynamic_v1_quint8",
            )

        assert any("encoder-model.static_qdq_v3.onnx" in p for p in session_paths)
        assert any("decoder_step.dynamic_v1_quint8.onnx" in p for p in session_paths)
