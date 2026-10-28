"""Checkpoint loading and frame-conversion discovery for the nemo ASR backend.

Covers :func:`coro.backends.asr.nemo_checkpoint.resolve_frame_conversion_factors`
and :func:`coro.backends.asr.nemo.build_nemo_asr_adapter`'s checkpoint/device
wiring. ``nemo`` and ``torch`` are replaced by module doubles; no NeMo
checkpoint is downloaded.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from coro.backends.asr.nemo import NemoASRAdapter, build_nemo_asr_adapter
from coro.backends.asr.nemo_checkpoint import resolve_frame_conversion_factors

_PROMPT_DICTIONARY = {"es-US": 0, "es-ES": 1, "en-US": 2}

# SentencePiece pieces for the fake hypothesis below.
_PIECES = {11: "▁hola", 12: ",", 13: "▁mundo", 14: "."}


class _FakeTokenizer:
    def ids_to_tokens(self, ids):
        return [_PIECES[i] for i in ids]


class TestResolveFrameConversionFactors:
    def test_reads_window_stride_and_encoder_subsampling_factor(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
            encoder=SimpleNamespace(subsampling_factor=8),
        )
        assert resolve_frame_conversion_factors(model) == (0.01, 8)

    def test_returns_none_when_window_stride_is_missing(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace()),
            encoder=SimpleNamespace(subsampling_factor=8),
        )
        assert resolve_frame_conversion_factors(model) is None

    def test_returns_none_when_encoder_has_no_subsampling_factor(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
            encoder=SimpleNamespace(),
        )
        assert resolve_frame_conversion_factors(model) is None

    def test_returns_none_when_model_has_no_encoder(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01))
        )
        assert resolve_frame_conversion_factors(model) is None


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

        with _build_fake_asr_model(fake_model):
            adapter = build_nemo_asr_adapter("nvidia/parakeet-rnnt-1.1b-prompt", device="cpu")

        assert adapter.frame_conversion == (0.01, 8)

    def test_frame_conversion_is_none_when_checkpoint_lacks_the_attributes(self):
        fake_model = SimpleNamespace(
            cfg={"model_defaults": {"prompt_dictionary": dict(_PROMPT_DICTIONARY)}},
            tokenizer=_FakeTokenizer(),
        )

        with _build_fake_asr_model(fake_model):
            adapter = build_nemo_asr_adapter("nvidia/parakeet-rnnt-1.1b-prompt", device="cpu")

        assert adapter.frame_conversion is None
