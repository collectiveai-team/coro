"""Checkpoint loading and frame-conversion discovery for the nemo ASR backend.

Covers :func:`coro.backends.asr.nemo_checkpoint.resolve_frame_conversion_factors`
and :func:`coro.backends.asr.nemo.build_nemo_asr_adapter`'s checkpoint/device
wiring. The ``nemo`` and ``torch`` modules are replaced by stand-ins whose
loaders return the in-memory checkpoint of ``support.fake_nemo_asr``; no NeMo
checkpoint is downloaded.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from support.fake_nemo_asr import FakeNemoPromptModel

from coro.backends.asr.nemo import NemoASRAdapter, build_nemo_asr_adapter
from coro.backends.asr.nemo_checkpoint import resolve_frame_conversion_factors

_PROMPT_DICTIONARY = {"es-US": 0, "es-ES": 1, "en-US": 2}
_CHECKPOINT = "nvidia/parakeet-rnnt-1.1b-prompt"


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


@pytest.fixture()
def load_checkpoint(monkeypatch):
    """Make ``build_nemo_asr_adapter`` load ``model`` for any checkpoint name, on a CPU host."""

    def _install(model: FakeNemoPromptModel) -> None:
        asr_mod = SimpleNamespace(
            models=SimpleNamespace(
                ASRModel=SimpleNamespace(
                    from_pretrained=lambda name: model,
                    restore_from=lambda path: model,
                )
            )
        )
        coll_mod = SimpleNamespace(asr=asr_mod)
        torch_mod = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        monkeypatch.setitem(sys.modules, "nemo", SimpleNamespace(collections=coll_mod))
        monkeypatch.setitem(sys.modules, "nemo.collections", coll_mod)
        monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_mod)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

    return _install


def _cfg_with_prompt_dictionary(**fields):
    defaults = {"model_defaults": {"prompt_dictionary": dict(_PROMPT_DICTIONARY)}}
    return SimpleNamespace(get=lambda key, default=None: defaults.get(key, default), **fields)


class TestBuildNemoAsrAdapter:
    async def test_builds_a_serialised_adapter_that_forces_the_checkpoint_languages(
        self, load_checkpoint
    ):
        """The adapter decodes with the checkpoint's own ``model.tokenizer``:
        on a real SentencePieceTokenizer, ``model.tokenizer.tokenizer`` has no
        ``ids_to_tokens`` (see build_nemo_asr_adapter)."""
        load_checkpoint(FakeNemoPromptModel(cfg=_cfg_with_prompt_dictionary()))

        adapter = build_nemo_asr_adapter(_CHECKPOINT, device="cpu")
        tokens = await adapter.transcribe_pcm(b"\x00\x00" * 16000, language="en")

        assert isinstance(adapter, NemoASRAdapter)
        assert adapter.admission.max_concurrency == 1
        assert [t.text for t in tokens] == [" hello", " world"]

    def test_resolves_frame_conversion_when_cfg_and_encoder_expose_it(self, load_checkpoint):
        load_checkpoint(
            FakeNemoPromptModel(
                cfg=_cfg_with_prompt_dictionary(preprocessor=SimpleNamespace(window_stride=0.01)),
                encoder=SimpleNamespace(subsampling_factor=8),
            )
        )

        adapter = build_nemo_asr_adapter(_CHECKPOINT, device="cpu")

        assert adapter.frame_conversion == (0.01, 8)

    def test_frame_conversion_is_none_when_checkpoint_lacks_the_attributes(self, load_checkpoint):
        load_checkpoint(FakeNemoPromptModel(cfg=_cfg_with_prompt_dictionary()))

        adapter = build_nemo_asr_adapter(_CHECKPOINT, device="cpu")

        assert adapter.frame_conversion is None
