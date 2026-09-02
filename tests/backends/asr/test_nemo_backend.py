"""NeMo ASR backend adapter.

The model is faked throughout: these tests cover the language ->
``target_lang`` resolution against the checkpoint's prompt dictionary, PCM ->
word-token conversion (SentencePiece grouping, end-time synthesis, absent
probability), the serialised Adapter Concurrency Policy, and the builder's
checkpoint/device wiring. No NeMo checkpoint is downloaded.

The fake model's ``transcribe`` mirrors the real
``EncDecHybridRNNTCTCBPEModelWithPrompt.transcribe`` signature closely enough
for ``_resolve_override_config_type``'s introspection to work (a typed,
optional ``override_config`` parameter) -- this is what makes the
forced-language tests below meaningful: they exercise the same code path
that discovered, against the real checkpoint, that a bare ``target_lang``
kwarg is silently ignored (see ``coro/backends/asr/nemo.py``'s module
docstring and ``.scratch/issue-64-language-constrained-asr/findings.md``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coro.backends.asr.nemo import (
    NemoASRAdapter,
    _build_forced_language_config,
    _frame_timestamps_to_seconds,
    _resolve_frame_conversion_factors,
    _resolve_override_config_type,
    _tokens_from_hypothesis,
    resolve_target_language,
)

# (window_stride, subsampling_factor) verified against the real checkpoint:
# 5.85s clip, frame indices [7, 9, ..., 72] -> seconds [0.56, 0.72, ..., 5.76].
_REAL_FRAME_CONVERSION = (0.01, 8)

_SAMPLE_RATE = 16000

_PROMPT_DICTIONARY = {"es-US": 0, "es-ES": 1, "en-US": 2}

# SentencePiece pieces for the fake hypothesis below.
_PIECES = {11: "▁hola", 12: ",", 13: "▁mundo", 14: "."}


class _FakeTokenizer:
    def ids_to_tokens(self, ids):
        return [_PIECES[i] for i in ids]


@dataclass
class _FakeTranscribeConfig:
    """Mirrors the fields of NeMo's real (base + Prompt-subclass) dataclass."""

    use_lhotse: bool = True
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: int | None = None
    timestamps: bool | None = None
    verbose: bool = True
    target_lang: str = "en-US"


class _FakePromptModel:
    """Records transcribe kwargs; returns one timestamped hypothesis."""

    def __init__(self, *, timestamps=True):
        self.calls: list[dict] = []
        self.written: tuple[int, int] | None = None
        self._timestamps = timestamps
        self.hypothesis = SimpleNamespace(
            y_sequence=(11, 12, 13, 14),  # real NeMo Hypothesis field name
            timestamp=[0.1, 0.2, 0.5, 0.6] if timestamps else None,
            text="hola, mundo.",
        )

    def transcribe(self, paths, *, override_config: _FakeTranscribeConfig | None = None, **kwargs):
        import soundfile as sf

        info = sf.info(paths[0])
        self.written = (int(info.samplerate), int(info.frames))
        call: dict = {"paths": list(paths), **kwargs}
        if override_config is not None:
            call["override_config"] = override_config
        self.calls.append(call)
        return [self.hypothesis]


class _FakeNoOverrideConfigModel:
    """A model whose ``transcribe`` has no typed ``override_config`` param.

    Represents a plain (non-prompt) checkpoint, or an unusual Prompt subclass
    that doesn't declare the parameter -- ``_build_forced_language_config``
    must raise loudly rather than silently drop the language request.
    """

    def transcribe(self, paths, **kwargs):
        raise AssertionError("should not be called: override_config resolution must fail first")


def _pcm(seconds: float = 1.0) -> bytes:
    return b"\x00\x00" * int(_SAMPLE_RATE * seconds)


async def _transcribe(model, language=None, *, prompt_dictionary=_PROMPT_DICTIONARY):
    adapter = NemoASRAdapter(model, tokenizer=_FakeTokenizer(), prompt_dictionary=prompt_dictionary)
    return await adapter.transcribe_pcm(_pcm(), language=language)


class TestResolveTargetLanguage:
    def test_exact_key_is_returned_unchanged(self):
        assert resolve_target_language("es-ES", _PROMPT_DICTIONARY) == "es-ES"

    def test_primary_subtag_resolves_to_first_matching_key(self):
        assert resolve_target_language("es", _PROMPT_DICTIONARY) == "es-US"

    def test_none_language_keeps_model_default(self):
        assert resolve_target_language(None, _PROMPT_DICTIONARY) is None

    def test_unknown_language_lists_supported_keys(self):
        with pytest.raises(ValueError, match=r"fr.*es-US"):
            resolve_target_language("fr", _PROMPT_DICTIONARY)


class TestNemoASRAdapter:
    async def test_language_is_forced_via_override_config(self):
        """A bare target_lang kwarg is silently ignored by NeMo's Lhotse-backed
        default dataloader (verified against a real checkpoint) -- language
        forcing must go through override_config with use_lhotse=False, or the
        request has no effect at all. num_workers=0 avoids a DataLoader
        multiprocessing crash (AF_UNIX path too long) under long checkout
        paths.
        """
        model = _FakePromptModel()
        await _transcribe(model, "es-US")
        call = model.calls[0]
        assert "target_lang" not in call  # never a bare kwarg
        override_config = call["override_config"]
        assert override_config.target_lang == "es-US"
        assert override_config.use_lhotse is False
        assert override_config.num_workers == 0

    async def test_forced_language_requests_no_timestamps(self):
        """timestamps=True crashes downstream (process_timestamp_outputs)
        on the forced-language / use_lhotse=False path -- a NeMo bug, not
        yet root-caused. Locks in the timestamps=False workaround so a NeMo
        upgrade that fixes the crash is a deliberate change, not an
        accidental regression back into it.

        timestamps=False must be set BOTH as a bare kwarg and inside
        override_config: transcribe()'s decoding-strategy reset
        (compute_timestamps/preserve_alignments) is gated on the top-level
        `timestamps is not None` check, evaluated before override_config is
        consulted at all -- override_config.timestamps alone leaves stale
        decoding state in place and hyp.timestamp comes back as a raw
        multi-element Tensor (crashing `getattr(hyp, "timestamp", None) or
        []`), verified against the real checkpoint.
        """
        model = _FakePromptModel()
        await _transcribe(model, "es-US")
        call = model.calls[0]
        assert call["override_config"].timestamps is False
        assert call["timestamps"] is False

    async def test_none_language_uses_plain_kwargs_not_override_config(self):
        model = _FakePromptModel()
        await _transcribe(model, None)
        call = model.calls[0]
        assert "target_lang" not in call
        assert "override_config" not in call
        assert call["timestamps"] is True

    async def test_forced_language_converts_frame_indices_to_seconds(self):
        """Verified against the real checkpoint: even with timestamps=False
        requested at both levels, hyp.timestamp still comes back as a raw
        per-token frame-index sequence (not seconds) -- when the
        checkpoint's window_stride/subsampling_factor are resolvable
        (frame_conversion is not None), the forced-language path now
        recovers true per-word timing from those frame indices instead of
        discarding them, using the same formula NeMo's own
        timestamp_utils.process_timestamp applies internally.
        """
        model = _FakePromptModel()
        model.hypothesis.timestamp = [7, 9, 10, 11]  # real frame indices, see findings.md
        adapter = NemoASRAdapter(
            model,
            tokenizer=_FakeTokenizer(),
            prompt_dictionary=_PROMPT_DICTIONARY,
            frame_conversion=_REAL_FRAME_CONVERSION,
        )
        tokens = await adapter.transcribe_pcm(_pcm(), language="es-US")
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        # word starts = first subword's converted time (frame_idx * 0.08):
        # "hola," starts at frame 7 -> 0.56s, "mundo." at frame 10 -> 0.8s.
        assert tokens[0].start == pytest.approx(0.56, abs=1e-9)
        assert tokens[1].start == pytest.approx(0.8, abs=1e-9)

    async def test_forced_language_falls_back_to_text_timing_without_frame_conversion(self):
        """A checkpoint whose window_stride/subsampling_factor could not be
        resolved at build time (frame_conversion=None, the adapter's
        default) must not have its raw frame indices misread as seconds --
        falls back to the same evenly-spaced text timing as the
        no-timestamps case, not (silently wrong) per-token grouping.
        """
        model = _FakePromptModel()
        model.hypothesis.timestamp = [7, 9, 10, 11]  # real frame indices, see findings.md
        tokens = await _transcribe(model, "es-US")  # default adapter: frame_conversion=None
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert tokens[0].start == 0.0
        assert tokens[-1].end <= 1.0

    async def test_language_without_prompt_dictionary_raises(self):
        model = _FakePromptModel()
        with pytest.raises(ValueError, match="no prompt dictionary"):
            await _transcribe(model, "es", prompt_dictionary=None)

    async def test_unknown_language_raises_from_the_adapter(self):
        model = _FakePromptModel()
        with pytest.raises(ValueError, match="not supported"):
            await _transcribe(model, "fr")

    async def test_subword_pieces_reconstruct_word_tokens(self):
        tokens = await _transcribe(_FakePromptModel())
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert [t.start for t in tokens] == [0.1, 0.5]
        assert [t.end for t in tokens] == [0.5, 0.7]
        assert all(t.probability is None for t in tokens)

    async def test_missing_timestamps_spread_over_the_clip(self):
        tokens = await _transcribe(_FakePromptModel(timestamps=False))
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert tokens[0].start == 0.0
        assert tokens[-1].end <= 1.0
        assert tokens == sorted(tokens, key=lambda t: t.start)

    async def test_empty_hypothesis_yields_no_tokens(self):
        model = _FakePromptModel()
        model.hypothesis = SimpleNamespace(y=(), timestamp=[], text="")
        assert await _transcribe(model) == []

    async def test_audio_is_written_as_16khz_wav(self):
        model = _FakePromptModel()
        await _transcribe(model)
        assert model.written is not None
        samplerate, frames = model.written
        assert samplerate == _SAMPLE_RATE
        assert frames == _SAMPLE_RATE

    def test_admission_is_serialised_to_one_permit(self):
        adapter = NemoASRAdapter(_FakePromptModel())
        assert adapter.admission.max_concurrency == 1


class _FakeTensorTimestamp:
    """Duck-types a torch.Tensor closely enough to trip the guard in
    ``_tokens_from_hypothesis`` (has ``.numel()``) without a torch dependency
    in this test module. Deliberately not iterable -- the guard must
    short-circuit on the ``numel`` check alone, matching the real bug: the
    original code crashed evaluating ``X or []`` truthiness *before* ever
    reaching iteration.
    """

    def numel(self) -> int:
        return 37


class TestTokensFromHypothesis:
    def test_tensor_timestamp_is_discarded_not_treated_as_seconds(self):
        """A raw per-token frame-index Tensor (verified shape from the real
        checkpoint) must never be treated as flat per-token seconds -- that
        would silently produce garbled word timing. Falls back to
        evenly-spaced text timing instead.
        """
        hyp = SimpleNamespace(
            y_sequence=(11, 12, 13, 14),
            timestamp=_FakeTensorTimestamp(),
            text="hola, mundo.",
        )
        tokens = _tokens_from_hypothesis(hyp, _FakeTokenizer(), span_end=1.0)
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert tokens[0].start == 0.0

    def test_prefers_y_sequence_over_legacy_y_attribute(self):
        """y_sequence is the real NeMo Hypothesis field; a stale/wrong ``y``
        value (99s aren't in the fake tokenizer's piece table) proves
        ``y_sequence`` -- not ``y`` -- is what actually gets read.
        """
        hyp = SimpleNamespace(
            y_sequence=(11, 12, 13, 14),
            y=(99, 99, 99, 99),
            timestamp=[0.1, 0.2, 0.5, 0.6],
            text="hola, mundo.",
        )
        tokens = _tokens_from_hypothesis(hyp, _FakeTokenizer(), span_end=1.0)
        assert [t.text for t in tokens] == [" hola,", " mundo."]

    def test_falls_back_to_legacy_y_attribute_when_y_sequence_absent(self):
        hyp = SimpleNamespace(
            y=(11, 12, 13, 14), timestamp=[0.1, 0.2, 0.5, 0.6], text="hola, mundo."
        )
        tokens = _tokens_from_hypothesis(hyp, _FakeTokenizer(), span_end=1.0)
        assert [t.text for t in tokens] == [" hola,", " mundo."]


class TestFrameTimestampsToSeconds:
    def test_converts_frame_indices_using_window_stride_and_subsampling(self):
        """Matches values verified directly against the real checkpoint."""
        seconds = _frame_timestamps_to_seconds([7, 9, 10, 72], _REAL_FRAME_CONVERSION)
        assert seconds == pytest.approx([0.56, 0.72, 0.8, 5.76], abs=1e-9)

    def test_returns_none_when_raw_timestamp_is_none(self):
        assert _frame_timestamps_to_seconds(None, _REAL_FRAME_CONVERSION) is None

    def test_returns_none_when_frame_conversion_is_none(self):
        assert _frame_timestamps_to_seconds([7, 9], None) is None

    def test_unwraps_tensor_like_via_tolist(self):
        class _FakeTensor:
            def tolist(self):
                return [7, 9]

        seconds = _frame_timestamps_to_seconds(_FakeTensor(), _REAL_FRAME_CONVERSION)
        assert seconds == pytest.approx([0.56, 0.72], abs=1e-9)

    def test_discards_structured_dict_shape(self):
        """The auto-detection path's {"word": ..., "char": ..., "segment":
        ...} shape must never be misread as a flat frame-index sequence.
        """
        assert _frame_timestamps_to_seconds({"word": []}, _REAL_FRAME_CONVERSION) is None


class TestResolveFrameConversionFactors:
    def test_reads_window_stride_and_encoder_subsampling_factor(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
            encoder=SimpleNamespace(subsampling_factor=8),
        )
        assert _resolve_frame_conversion_factors(model) == (0.01, 8)

    def test_returns_none_when_window_stride_is_missing(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace()),
            encoder=SimpleNamespace(subsampling_factor=8),
        )
        assert _resolve_frame_conversion_factors(model) is None

    def test_returns_none_when_encoder_has_no_subsampling_factor(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
            encoder=SimpleNamespace(),
        )
        assert _resolve_frame_conversion_factors(model) is None

    def test_returns_none_when_model_has_no_encoder(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01))
        )
        assert _resolve_frame_conversion_factors(model) is None


class TestResolveOverrideConfigType:
    def test_discovers_type_from_typed_optional_parameter(self):
        assert _resolve_override_config_type(_FakePromptModel()) is _FakeTranscribeConfig

    def test_returns_none_when_parameter_is_untyped(self):
        assert _resolve_override_config_type(_FakeNoOverrideConfigModel()) is None


class TestBuildForcedLanguageConfig:
    def test_forces_use_lhotse_false_and_zero_workers(self):
        config = _build_forced_language_config(_FakePromptModel(), "es-US")
        assert isinstance(config, _FakeTranscribeConfig)
        assert config.target_lang == "es-US"
        assert config.use_lhotse is False
        assert config.num_workers == 0
        assert config.timestamps is False
        assert config.return_hypotheses is True

    def test_raises_loudly_when_no_override_config_type_is_discoverable(self):
        with pytest.raises(ValueError, match="override_config"):
            _build_forced_language_config(_FakeNoOverrideConfigModel(), "es")


def _build_fake_asr_model(fake_model):
    """Wire a fake NeMo model behind the module-patching build_nemo_asr_adapter needs."""
    fake_model.eval = lambda: fake_model
    fake_model.to = lambda device: fake_model
    asr_mod = SimpleNamespace(
        models=SimpleNamespace(
            ASRModel=SimpleNamespace(
                from_pretrained=lambda name: fake_model,
                restore_from=lambda path: fake_model,
            )
        )
    )
    coll_mod = SimpleNamespace(asr=asr_mod)
    nemo_mod = SimpleNamespace(collections=coll_mod)
    torch_mod = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    return patch.dict(
        sys.modules,
        {
            "nemo": nemo_mod,
            "nemo.collections": coll_mod,
            "nemo.collections.asr": asr_mod,
            "torch": torch_mod,
        },
    )


class TestBuildNemoAsrAdapter:
    def test_builds_adapter_with_checkpoint_metadata(self):
        # model.tokenizer is used directly (not unwrapped via a `.tokenizer`
        # inner attribute) -- see the module docstring's note on why
        # unwrapping reaches into an implementation-specific inner object
        # lacking `ids_to_tokens` on a real SentencePieceTokenizer checkpoint.
        fake_model = SimpleNamespace(
            cfg={"model_defaults": {"prompt_dictionary": dict(_PROMPT_DICTIONARY)}},
            tokenizer=_FakeTokenizer(),
        )

        from coro.backends.asr.nemo import build_nemo_asr_adapter

        with _build_fake_asr_model(fake_model):
            adapter = build_nemo_asr_adapter("nvidia/parakeet-rnnt-1.1b-prompt", device="cpu")

        assert isinstance(adapter, NemoASRAdapter)
        assert adapter.admission.max_concurrency == 1
        assert adapter._tokenizer is fake_model.tokenizer

    def test_resolves_frame_conversion_when_cfg_and_encoder_expose_it(self):
        fake_model = SimpleNamespace(
            cfg=SimpleNamespace(
                get=lambda key, default=None: {
                    "model_defaults": {"prompt_dictionary": dict(_PROMPT_DICTIONARY)}
                }.get(key, default),
                preprocessor=SimpleNamespace(window_stride=0.01),
            ),
            encoder=SimpleNamespace(subsampling_factor=8),
            tokenizer=_FakeTokenizer(),
        )

        from coro.backends.asr.nemo import build_nemo_asr_adapter

        with _build_fake_asr_model(fake_model):
            adapter = build_nemo_asr_adapter("nvidia/parakeet-rnnt-1.1b-prompt", device="cpu")

        assert adapter.frame_conversion == (0.01, 8)

    def test_frame_conversion_is_none_when_checkpoint_lacks_the_attributes(self):
        fake_model = SimpleNamespace(
            cfg={"model_defaults": {"prompt_dictionary": dict(_PROMPT_DICTIONARY)}},
            tokenizer=_FakeTokenizer(),
        )

        from coro.backends.asr.nemo import build_nemo_asr_adapter

        with _build_fake_asr_model(fake_model):
            adapter = build_nemo_asr_adapter("nvidia/parakeet-rnnt-1.1b-prompt", device="cpu")

        assert adapter.frame_conversion is None
