"""Behaviour of the faster-whisper adapter's language handling.

The fake model below builds faster-whisper's own ``Tokenizer`` the way
``WhisperModel.transcribe`` does, so the ``ValueError`` an unknown language
produces is the real one rather than an imagined stand-in.
"""

from __future__ import annotations

import pytest
from faster_whisper.tokenizer import Tokenizer

from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.faster_whisper import FasterWhisperASRAdapter

_PCM = b"\x00\x00" * 1600


class _StubTokenizer:
    def token_to_id(self, _token: str) -> int:
        return 1


class _FakeWhisperModel:
    """Mimics ``WhisperModel.transcribe`` validating the language eagerly."""

    supported_languages = ("en", "es", "fr")

    def transcribe(self, audio, *, language=None, **_kwargs):
        Tokenizer(_StubTokenizer(), True, task="transcribe", language=language or "en")
        return iter([]), None


class TestForcedLanguage:
    async def test_unknown_language_is_an_unsupported_language(self):
        """A client-fixable request, so it must reach the API as a 400, not a 500."""
        adapter = FasterWhisperASRAdapter(_FakeWhisperModel())

        with pytest.raises(AsrUnsupportedLanguageError) as excinfo:
            await adapter.transcribe_pcm(_PCM, language="xx")

        assert excinfo.value.language == "xx"
        assert set(excinfo.value.supported_languages) == {"en", "es", "fr"}

    async def test_supported_language_passes_through(self):
        adapter = FasterWhisperASRAdapter(_FakeWhisperModel())

        assert await adapter.transcribe_pcm(_PCM, language="es") == []

    async def test_other_value_errors_stay_server_errors(self):
        """Only a language rejection is the client's fault; any other ValueError is ours."""

        class _BrokenModel(_FakeWhisperModel):
            def transcribe(self, audio, **_kwargs):
                msg = "corrupt model state"
                raise ValueError(msg)

        adapter = FasterWhisperASRAdapter(_BrokenModel())

        with pytest.raises(ValueError, match="corrupt model state") as excinfo:
            await adapter.transcribe_pcm(_PCM, language="es")

        assert not isinstance(excinfo.value, AsrUnsupportedLanguageError)
