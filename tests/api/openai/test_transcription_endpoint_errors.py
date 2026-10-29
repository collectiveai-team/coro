"""Transcription Endpoint mapping of pipeline and backend failures to OpenAI-style errors.

Each test injects a pipeline that fails the way a backend does and asserts the
status code and error body the public Transcription API Contract promises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from coro.api.exceptions import UNDECODABLE_MEDIA_MESSAGE
from coro.audio import AudioConversionError
from coro.backends.asr.concurrency import AsrCapacityError
from support.factories import make_app as _app_with_pipeline
from support.factories import make_wav as _minimal_wav_bytes


@pytest.mark.asyncio
async def test_transcription_endpoint_undecodable_media_returns_400():
    """An undecodable upload is a client error: 400 invalid_request_error, no ffmpeg leak."""

    class _BadMediaPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            raise AudioConversionError("Audio conversion failed: moov atom not found")

    app = _app_with_pipeline(_BadMediaPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.mp4", b"not really an mp4", "video/mp4")},
            data={"model": "whisper-1"},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "file"
    assert body["error"]["message"] == UNDECODABLE_MEDIA_MESSAGE
    assert "moov atom" not in body["error"]["message"]  # ffmpeg detail not leaked


@pytest.mark.asyncio
async def test_transcription_endpoint_internal_valueerror_stays_500():
    """A non-conversion ValueError (e.g. config) is NOT misclassified as a 400."""

    class _ConfigErrorPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            raise ValueError("overlap_seconds must be less than window_seconds")

    app = _app_with_pipeline(_ConfigErrorPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1"},
        )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"] == "Transcription processing failed."


@pytest.mark.asyncio
async def test_transcription_endpoint_sheds_load_at_asr_capacity():
    """Admission-control rejection is a 429 with a Retry-After hint, not a 500."""

    class _AtCapacityPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            raise AsrCapacityError("Server is at ASR capacity.", retry_after_seconds=5.0)

    app = _app_with_pipeline(_AtCapacityPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1"},
        )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    body = response.json()
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["error"]["code"] == "server_overloaded"
    assert body["error"]["message"] == "Server is at ASR capacity."


@pytest.mark.asyncio
async def test_transcription_endpoint_rejects_unsupported_language():
    """An unsupported Canary language is a 400 naming the supported set, not a 500."""
    from coro.backends.asr.errors import AsrUnsupportedLanguageError

    class _UnsupportedLanguagePipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            raise AsrUnsupportedLanguageError("ja", supported_languages=["en", "es", "fr"])

    app = _app_with_pipeline(_UnsupportedLanguagePipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1", "language": "ja"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "language"
    assert "ja" in body["error"]["message"]
    assert "en" in body["error"]["message"]


@pytest.mark.asyncio
async def test_transcription_endpoint_rejects_language_forcing_on_a_non_prompt_nemo_checkpoint():
    """The real nemo adapter, built on a checkpoint without a prompt dictionary, answers a
    forced language with a 400 naming ``language`` (was a generic 500)."""
    from coro.backends.asr.nemo import NemoASRAdapter

    # spec_set=[]: any attribute access fails, so the adapter must refuse the
    # language before it touches the model or the tokenizer.
    adapter = NemoASRAdapter(
        MagicMock(spec_set=[]), tokenizer=MagicMock(spec_set=[]), prompt_dictionary=None
    )

    class _NemoPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            return await adapter.transcribe_pcm(b"\x00\x00" * 1600, language=language)

    app = _app_with_pipeline(_NemoPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1", "language": "es"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "language"
    assert "'es'" in body["error"]["message"]


@pytest.mark.asyncio
async def test_transcription_endpoint_rejects_an_unknown_language_on_faster_whisper():
    """The real faster-whisper adapter answers a language the model rejects with a 400 naming
    ``language`` (was a generic 500 from faster-whisper's bare ``ValueError``)."""
    from faster_whisper.tokenizer import Tokenizer

    from coro.backends.asr.faster_whisper import FasterWhisperASRAdapter

    class _TokenizerStub:
        def token_to_id(self, _token):
            return 1

    class _WhisperModel:
        supported_languages = ("en", "es", "fr")

        def transcribe(self, audio, *, language=None, **_kwargs):
            Tokenizer(_TokenizerStub(), True, task="transcribe", language=language or "en")
            return iter([]), None

    adapter = FasterWhisperASRAdapter(_WhisperModel())

    class _FasterWhisperPipeline:
        async def transcribe(self, audio, *, language=None, prompt=None):
            return await adapter.transcribe_pcm(b"\x00\x00" * 1600, language=language)

    app = _app_with_pipeline(_FasterWhisperPipeline())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav_bytes(), "audio/wav")},
            data={"model": "whisper-1", "language": "xx"},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "language"
    assert "'xx'" in body["error"]["message"]
    assert "es" in body["error"]["message"]
