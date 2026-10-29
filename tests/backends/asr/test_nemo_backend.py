"""NeMo ASR backend adapter, over the in-memory checkpoint of ``support.fake_nemo_asr``.

These tests cover the language -> ``target_lang`` resolution against the
checkpoint's prompt dictionary, PCM -> word-token conversion (SentencePiece
grouping, end-time synthesis, absent probability) and the serialised Adapter
Concurrency Policy. No NeMo checkpoint is downloaded.

The fake checkpoint keeps the NeMo quirks found against the real one: a bare
``target_lang`` keyword is ignored, and the forced-language path crashes
unless ``timestamps=False`` is passed at both levels (see the module
docstrings of ``coro/backends/asr/nemo.py`` and ``support/fake_nemo_asr.py``).
A forced-language test that gets the expected words back has therefore gone
through the path that works on the real checkpoint.
"""

from __future__ import annotations

import pytest
from support.fake_nemo_asr import (
    SILENCE,
    FakeNemoPromptModel,
    FakeSubwordTokenizer,
    FakeTranscribeConfig,
    HeardAudio,
)

from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.nemo import NemoASRAdapter
from coro.backends.asr.nemo_checkpoint import (
    build_forced_language_config,
    resolve_override_config_type,
    resolve_target_language,
)

# (window_stride, subsampling_factor) verified against the real checkpoint:
# 5.85s clip, frame indices [7, 9, ..., 72] -> seconds [0.56, 0.72, ..., 5.76].
REAL_FRAME_CONVERSION = (0.01, 8)

SAMPLE_RATE = 16000

PROMPT_DICTIONARY = {"es-US": 0, "es-ES": 1, "en-US": 2}


class NoOverrideConfigModel:
    """A model whose ``transcribe`` has no typed ``override_config`` param.

    Represents a plain (non-prompt) checkpoint, or an unusual Prompt subclass
    that doesn't declare the parameter -- ``build_forced_language_config``
    must raise loudly rather than silently drop the language request.
    """

    def transcribe(self, paths, **kwargs):
        raise AssertionError("should not be called: override_config resolution must fail first")


def silence(seconds: float = 1.0) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


def make_adapter(model, *, prompt_dictionary=PROMPT_DICTIONARY, frame_conversion=None):
    return NemoASRAdapter(
        model,
        tokenizer=FakeSubwordTokenizer(),
        prompt_dictionary=prompt_dictionary,
        frame_conversion=frame_conversion,
    )


async def transcribe(model, language=None, *, prompt_dictionary=PROMPT_DICTIONARY):
    adapter = make_adapter(model, prompt_dictionary=prompt_dictionary)
    return await adapter.transcribe_pcm(silence(), language=language)


class TestResolveTargetLanguage:
    def test_exact_key_is_returned_unchanged(self):
        assert resolve_target_language("es-ES", PROMPT_DICTIONARY) == "es-ES"

    def test_primary_subtag_resolves_to_first_matching_key(self):
        assert resolve_target_language("es", PROMPT_DICTIONARY) == "es-US"

    def test_none_language_keeps_model_default(self):
        assert resolve_target_language(None, PROMPT_DICTIONARY) is None

    def test_unknown_language_lists_supported_keys(self):
        with pytest.raises(AsrUnsupportedLanguageError, match=r"fr.*es-US") as excinfo:
            resolve_target_language("fr", PROMPT_DICTIONARY)
        assert excinfo.value.language == "fr"
        assert set(excinfo.value.supported_languages) == set(PROMPT_DICTIONARY)


class TestNemoASRAdapter:
    async def test_requested_language_is_the_language_transcribed(self):
        """NeMo drops a bare target_lang; only an override_config with
        use_lhotse=False, timestamps=False at both levels and num_workers=0
        forces the language without being ignored or crashing."""
        tokens = await transcribe(FakeNemoPromptModel(), "en")

        assert [t.text for t in tokens] == [" hello", " world"]

    async def test_no_language_keeps_the_checkpoint_default_and_its_timestamps(self):
        tokens = await transcribe(FakeNemoPromptModel(default_language="en-US"))

        assert [t.text for t in tokens] == [" hello", " world"]
        assert [t.start for t in tokens] == [0.1, 0.5]

    async def test_forced_language_converts_frame_indices_to_seconds(self):
        """The forced-language path returns raw per-token frame indices, not
        seconds, even with timestamps=False. With the checkpoint's
        window_stride/subsampling_factor known, the adapter recovers per-word
        timing with the formula of NeMo's timestamp_utils.process_timestamp.
        """
        adapter = make_adapter(FakeNemoPromptModel(), frame_conversion=REAL_FRAME_CONVERSION)

        tokens = await adapter.transcribe_pcm(silence(), language="en-US")

        assert [t.text for t in tokens] == [" hello", " world"]
        # Frames 7 and 10 times 0.01 s * 8.
        assert tokens[0].start == pytest.approx(0.56, abs=1e-9)
        assert tokens[1].start == pytest.approx(0.8, abs=1e-9)

    async def test_forced_language_falls_back_to_text_timing_without_frame_conversion(self):
        """Raw frame indices must not be misread as seconds when the conversion
        factors are unknown: timing falls back to evenly spaced text timing.
        """
        tokens = await transcribe(FakeNemoPromptModel(), "es-US")

        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert tokens[0].start == 0.0
        assert tokens[-1].end <= 1.0

    async def test_language_without_prompt_dictionary_is_an_unsupported_language(self):
        """A client-fixable request (drop ``language``), so it must reach the API as a 400."""
        with pytest.raises(AsrUnsupportedLanguageError, match="not supported") as excinfo:
            await transcribe(FakeNemoPromptModel(), "es", prompt_dictionary=None)
        assert excinfo.value.language == "es"
        assert excinfo.value.supported_languages == ()

    async def test_unknown_language_raises_from_the_adapter(self):
        with pytest.raises(AsrUnsupportedLanguageError, match="not supported"):
            await transcribe(FakeNemoPromptModel(), "fr")

    async def test_subword_pieces_reconstruct_word_tokens(self):
        tokens = await transcribe(FakeNemoPromptModel())
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert [t.start for t in tokens] == [0.1, 0.5]
        assert [t.end for t in tokens] == [0.5, 0.7]
        assert all(t.probability is None for t in tokens)

    async def test_missing_timestamps_spread_over_the_clip(self):
        tokens = await transcribe(FakeNemoPromptModel(emits_timestamps=False))
        assert [t.text for t in tokens] == [" hola,", " mundo."]
        assert tokens[0].start == 0.0
        assert tokens[-1].end <= 1.0
        assert tokens == sorted(tokens, key=lambda t: t.start)

    async def test_empty_hypothesis_yields_no_tokens(self):
        model = FakeNemoPromptModel(utterances={"es-US": SILENCE})
        assert await transcribe(model) == []

    async def test_audio_is_written_as_16khz_wav(self):
        model = FakeNemoPromptModel()
        await transcribe(model)
        assert model.heard == [HeardAudio(samplerate=SAMPLE_RATE, frames=SAMPLE_RATE)]

    def test_admission_is_serialised_to_one_permit(self):
        adapter = NemoASRAdapter(FakeNemoPromptModel())
        assert adapter.admission.max_concurrency == 1


class TestResolveOverrideConfigType:
    def test_discovers_type_from_typed_optional_parameter(self):
        assert resolve_override_config_type(FakeNemoPromptModel()) is FakeTranscribeConfig

    def test_returns_none_when_parameter_is_untyped(self):
        assert resolve_override_config_type(NoOverrideConfigModel()) is None


class TestBuildForcedLanguageConfig:
    def test_forces_use_lhotse_false_and_zero_workers(self):
        config = build_forced_language_config(FakeNemoPromptModel(), "es-US")
        assert isinstance(config, FakeTranscribeConfig)
        assert config.target_lang == "es-US"
        assert config.use_lhotse is False
        assert config.num_workers == 0
        assert config.timestamps is False
        assert config.return_hypotheses is True

    def test_raises_loudly_when_no_override_config_type_is_discoverable(self):
        with pytest.raises(ValueError, match="override_config") as excinfo:
            build_forced_language_config(NoOverrideConfigModel(), "es")
        # A checkpoint/NeMo API mismatch is the deployment's fault, not the
        # request's: it must stay a server error rather than a language 400.
        assert not isinstance(excinfo.value, AsrUnsupportedLanguageError)
