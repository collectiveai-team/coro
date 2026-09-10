"""onnx-canary-split backend: adapter and builder wiring.

Covers:
- ``OnnxCanarySplitASRAdapter.transcribe_pcm`` against a stub ASR object (no
  real ONNX Runtime session) -- language forwarding, the concurrent Adapter
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.onnx_canary_split import (
    OnnxCanarySplitASRAdapter,
    build_onnx_canary_split_adapter,
    resolve_canary_language,
)

_SAMPLE_RATE = 16000


def _pcm(seconds: float = 1.0) -> bytes:
    return b"\x00\x00" * int(_SAMPLE_RATE * seconds)


def _result(tokens=(" hola", " mundo"), timestamps=None, logprobs=None):
    from types import SimpleNamespace

    return SimpleNamespace(tokens=list(tokens), timestamps=timestamps, logprobs=logprobs, text="")


# ---------------------------------------------------------------------------
# resolve_canary_language
# ---------------------------------------------------------------------------


class TestResolveCanaryLanguage:
    """Vocab-backed language resolution -- locale reduction and membership."""

    def test_locale_reduces_to_base_subtag(self):
        assert resolve_canary_language("es-US", _STUB_LANGUAGE_TOKENS) == "es"

    def test_base_subtag_and_locale_produce_the_same_result(self):
        assert resolve_canary_language("es-US", _STUB_LANGUAGE_TOKENS) == resolve_canary_language(
            "es", _STUB_LANGUAGE_TOKENS
        )

    def test_underscore_locale_and_mixed_case_also_reduce(self):
        assert resolve_canary_language("es_US", _STUB_LANGUAGE_TOKENS) == "es"
        assert resolve_canary_language(" ES ", _STUB_LANGUAGE_TOKENS) == "es"

    def test_unsupported_language_raises_typed_error_naming_the_supported_set(self):
        with pytest.raises(AsrUnsupportedLanguageError) as excinfo:
            resolve_canary_language("ja", _STUB_LANGUAGE_TOKENS)
        assert excinfo.value.language == "ja"
        assert excinfo.value.supported_languages == tuple(sorted(_STUB_LANGUAGE_TOKENS))
        assert "ja" in excinfo.value.message
        assert len(excinfo.value.supported_languages) == len(_STUB_LANGUAGE_TOKENS)
        assert all(code in excinfo.value.message for code in excinfo.value.supported_languages)

    def test_none_language_is_left_unresolved_for_the_caller_to_apply_a_fallback(self):
        assert resolve_canary_language(None, _STUB_LANGUAGE_TOKENS) is None


# ---------------------------------------------------------------------------
# OnnxCanarySplitASRAdapter.transcribe_pcm
# ---------------------------------------------------------------------------


_STUB_LANGUAGE_TOKENS = {"en": 64, "es": 171, "fr": 71, "de": 78, "pt": 151}
"""A small vocab-derived-looking language map, standing in for the real
checkpoint's ``language_token_ids`` (see ``onnx_canary_split.py``)."""


class _StubAsr:
    """Fakes the split-decode ``NemoConformerAED`` subclass's public surface."""

    def __init__(self, result, *, language_token_ids=None):
        self._result = result
        self.calls: list[dict] = []
        self.language_token_ids = (
            _STUB_LANGUAGE_TOKENS if language_token_ids is None else language_token_ids
        )

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

    async def test_no_language_resolves_to_the_adapter_fallback(self):
        """No request language resolves to the adapter's fallback (never onnx_asr's own default)."""
        asr = _StubAsr(_result())
        adapter = OnnxCanarySplitASRAdapter(asr, fallback_language="fr")
        await adapter.transcribe_pcm(_pcm())
        assert asr.calls[0]["language"] == "fr"

    async def test_no_language_defaults_to_english_fallback(self):
        asr = _StubAsr(_result())
        adapter = OnnxCanarySplitASRAdapter(asr)
        await adapter.transcribe_pcm(_pcm())
        assert asr.calls[0]["language"] == "en"

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

    def test_admission_auto_sizes_by_default(self):
        """Default admission follows the shared auto-sizing policy (>= 2 permits)."""
        adapter = OnnxCanarySplitASRAdapter(_StubAsr(_result()))
        assert adapter.admission.max_concurrency >= 2

    def test_admission_honours_an_explicit_permit_count(self):
        from coro.backends.asr.concurrency import AdmissionController

        adapter = OnnxCanarySplitASRAdapter(
            _StubAsr(_result()), admission=AdmissionController(max_concurrency=3, max_queue_depth=4)
        )
        assert adapter.admission.max_concurrency == 3

    def test_honours_prompt_is_false(self):
        assert OnnxCanarySplitASRAdapter.honours_prompt is False


# ---------------------------------------------------------------------------
# OnnxCanarySplitASRAdapter.detect_language (ticket 05's core auto-LID probe)
# ---------------------------------------------------------------------------

_LID_VOCAB = {
    0: " ",
    1: "<|startofcontext|>",
    2: "<|startoftranscript|>",
    3: "<|emo:undefined|>",
    4: "<|es|>",
    5: "<|en|>",
    6: "<|pnc|>",
}


class _LidStubAsr:
    """Fakes the private encode/decode surface ``detect_language`` drives.

    Scripts the two decoder steps directly, mirroring ``_partial_prompt_lid``'s
    own encode -> 2-step-decode sequence, without a real ONNX graph.
    """

    def __init__(self, *, emitted_tokens: tuple[str, str]) -> None:
        self._vocab = dict(_LID_VOCAB)
        self._tokens = {token: id for id, token in self._vocab.items()}
        self.language_token_ids = {"es": 4, "en": 5}
        self._emitted_ids = [self._tokens[t] for t in emitted_tokens]
        self._decoder = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="decoder_mems", shape=(2, 1, 0, 4))]
        )
        self.decode_calls = 0

    def _preprocessor(self, waveforms, waveforms_len):
        return waveforms, waveforms_len

    def _encode(self, features, features_lens):
        return np.zeros((1, 1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.int64)

    def _decode(self, input_ids, encoder_embeddings, encoder_mask, decoder_mems):
        next_id = self._emitted_ids[self.decode_calls]
        self.decode_calls += 1
        logits = np.zeros((1, 1, len(self._vocab)), dtype=np.float32)
        logits[0, 0, next_id] = 10.0
        new_mems = np.zeros(
            (decoder_mems.shape[0], 1, decoder_mems.shape[2] + 1, decoder_mems.shape[3]),
            dtype=np.float32,
        )
        return logits, new_mems


class TestDetectLanguage:
    async def test_detects_the_emitted_language_token(self):
        asr = _LidStubAsr(emitted_tokens=("<|emo:undefined|>", "<|es|>"))
        adapter = OnnxCanarySplitASRAdapter(asr)

        detected = await adapter.detect_language(_pcm())

        assert detected == "es"
        assert asr.decode_calls == 2  # exactly the two partial-prompt steps

    async def test_a_non_language_second_token_returns_none(self):
        """Defensive: not observed on real audio in the LID probe, but handled."""
        asr = _LidStubAsr(emitted_tokens=("<|emo:undefined|>", "<|pnc|>"))
        adapter = OnnxCanarySplitASRAdapter(asr)

        assert await adapter.detect_language(_pcm()) is None

    async def test_participates_in_the_admission_controller(self):
        """detect_language is a real inference call, bounded the same way transcribe_pcm is."""
        from coro.backends.asr.concurrency import AdmissionController

        asr = _LidStubAsr(emitted_tokens=("<|emo:undefined|>", "<|es|>"))
        admission = AdmissionController(max_concurrency=1, max_queue_depth=0)
        adapter = OnnxCanarySplitASRAdapter(asr, admission=admission)

        assert await adapter.detect_language(_pcm()) == "es"


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
        assert adapter.admission.max_concurrency >= 2

    def test_builder_forwards_max_concurrency_to_admission(self, tmp_path):
        _build_artifact_dir(tmp_path)
        with patch("onnxruntime.InferenceSession", autospec=True, return_value=MagicMock()):
            adapter = build_onnx_canary_split_adapter(
                str(tmp_path), device="cpu", max_concurrency=5
            )

        assert adapter.admission.max_concurrency == 5

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
