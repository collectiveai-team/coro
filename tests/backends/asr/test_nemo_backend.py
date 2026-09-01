"""NeMo ASR backend adapter.

The model is faked throughout: these tests cover the language ->
``target_lang`` resolution against the checkpoint's prompt dictionary, PCM ->
word-token conversion (SentencePiece grouping, end-time synthesis, absent
probability), the serialised Adapter Concurrency Policy, and the builder's
checkpoint/device wiring. No NeMo checkpoint is downloaded.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coro.backends.asr.nemo import NemoASRAdapter, resolve_target_language

_SAMPLE_RATE = 16000

_PROMPT_DICTIONARY = {"es-US": 0, "es-ES": 1, "en-US": 2}

# SentencePiece pieces for the fake hypothesis below.
_PIECES = {11: "▁hola", 12: ",", 13: "▁mundo", 14: "."}


class _FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return [_PIECES[i] for i in ids]


class _FakePromptModel:
    """Records transcribe kwargs; returns one timestamped hypothesis."""

    def __init__(self, *, timestamps=True):
        self.calls: list[dict] = []
        self.written: tuple[int, int] | None = None
        self._timestamps = timestamps
        self.hypothesis = SimpleNamespace(
            y=(11, 12, 13, 14),
            timestamp=[0.1, 0.2, 0.5, 0.6] if timestamps else None,
            text="hola, mundo.",
        )

    def transcribe(self, paths, **kwargs):
        import soundfile as sf

        info = sf.info(paths[0])
        self.written = (int(info.samplerate), int(info.frames))
        self.calls.append({"paths": list(paths), **kwargs})
        return [self.hypothesis]


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
    async def test_language_is_passed_as_target_lang(self):
        model = _FakePromptModel()
        await _transcribe(model, "es-US")
        assert model.calls[0]["target_lang"] == "es-US"

    async def test_none_language_omits_target_lang(self):
        model = _FakePromptModel()
        await _transcribe(model, None)
        assert "target_lang" not in model.calls[0]

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


class TestBuildNemoAsrAdapter:
    def test_builds_adapter_with_checkpoint_metadata(self):
        fake_model = SimpleNamespace(
            cfg={"model_defaults": {"prompt_dictionary": dict(_PROMPT_DICTIONARY)}},
            tokenizer=SimpleNamespace(tokenizer=_FakeTokenizer()),
        )
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

        from coro.backends.asr.nemo import build_nemo_asr_adapter

        with patch.dict(
            sys.modules,
            {
                "nemo": nemo_mod,
                "nemo.collections": coll_mod,
                "nemo.collections.asr": asr_mod,
                "torch": torch_mod,
            },
        ):
            adapter = build_nemo_asr_adapter("nvidia/parakeet-rnnt-1.1b-prompt", device="cpu")

        assert isinstance(adapter, NemoASRAdapter)
        assert adapter.admission.max_concurrency == 1
