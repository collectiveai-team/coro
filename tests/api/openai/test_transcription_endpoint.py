"""Public Transcription Endpoint behavior.

Requests run through the real Full-Memory Pipeline, with ffmpeg decoding the
upload, and in-memory ASR and diarization adapters in place of the models.
All assertions target the public Transcription API Contract.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from coro.core.models import SpeakerSegment, TranscriptToken
from coro.pipelines.full_memory import FullMemoryPipeline
from support.adapters import FixedDiarization, RecordingASR
from support.factories import make_app, make_wav

ONE_SECOND_WAV = make_wav(frames=16000)


def _asr() -> RecordingASR:
    """An ASR Adapter that hears "hello world" in every window."""
    return RecordingASR(
        [
            TranscriptToken(start=0.0, end=0.4, text=" hello", probability=1.0),
            TranscriptToken(start=0.5, end=1.0, text=" world", probability=1.0),
        ]
    )


def _app(asr: RecordingASR | None = None):
    """Build an app whose pipeline transcribes with ``asr`` and finds one speaker."""
    diarization = FixedDiarization([SpeakerSegment(start=0.0, end=1.0, speaker=0)])
    return make_app(FullMemoryPipeline(asr=asr or _asr(), diarization=diarization))


async def _post(app, data: dict, *, upload: bytes = ONE_SECOND_WAV, filename: str = "test.wav"):
    """POST one transcription request to ``app``."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/v1/audio/transcriptions",
            files={"file": (filename, upload, "audio/wav")},
            data=data,
        )


@pytest.mark.asyncio
async def test_transcription_endpoint_returns_default_openai_json():
    """POST /v1/audio/transcriptions returns default OpenAI JSON."""
    response = await _post(_app(), {"model": "whisper-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["usage"] == {"type": "duration", "seconds": 1}


@pytest.mark.asyncio
async def test_transcription_endpoint_accepts_openai_compatible_form_params():
    """POST /v1/audio/transcriptions accepts OpenAI-compatible form fields."""
    response = await _post(
        _app(),
        {
            "model": "whisper-1",
            "language": "es",
            "prompt": "some hint",
            "response_format": "verbose_json",
            "temperature": "0",
            "timestamp_granularities[]": ["word", "segment"],
            "stream": "false",
        },
    )

    assert response.status_code == 200
    assert response.json()["text"] == "hello world"


@pytest.mark.asyncio
async def test_transcription_endpoint_empty_upload_returns_openai_style_error():
    """POST /v1/audio/transcriptions with empty body returns OpenAI-style error."""
    response = await _post(_app(), {"model": "whisper-1"}, upload=b"", filename="empty.wav")

    assert response.status_code == 400
    body = response.json()
    # OpenAI-style error: {"error": {"message": "...", "type": "...", ...}}
    assert "error" in body
    assert "message" in body["error"]


@pytest.mark.asyncio
async def test_transcription_endpoint_rejects_invalid_language_tag():
    """A non-BCP-47 language hint returns a 400, not a generic 500."""
    response = await _post(_app(), {"model": "whisper-1", "language": "string"})

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "language"


@pytest.mark.asyncio
async def test_transcription_endpoint_passes_the_canonical_language_to_the_pipeline():
    """Equivalent spellings of a language reach the ASR Adapter in one BCP-47 form."""
    asr = _asr()
    app = _app(asr)
    statuses = [
        (await _post(app, {"model": "whisper-1", "language": spelling})).status_code
        for spelling in ("ES", "es_ar", " es-US ")
    ]

    assert statuses == [200, 200, 200]
    assert asr.languages == ["es", "es-AR", "es-US"]


@pytest.mark.asyncio
async def test_transcription_endpoint_treats_blank_language_as_unset():
    """An empty / whitespace-only language hint is coerced to None and succeeds."""
    asr = _asr()
    app = _app(asr)
    statuses = [
        (await _post(app, {"model": "whisper-1", "language": blank})).status_code
        for blank in ("", "   ")
    ]

    assert statuses == [200, 200]
    assert asr.languages == [None, None]


@pytest.mark.asyncio
async def test_transcription_endpoint_tolerates_swagger_blank_fields():
    """Swagger's 'Try it out' sends every optional field as an empty string.

    The empty ``known_speaker_references[]`` value must not fail UploadFile
    parsing with a 422; blanks are treated as unset.
    """
    response = await _post(
        _app(),
        {
            "stream": "",
            "prompt": "",
            "timestamp_granularities[]": "",
            "known_speaker_references[]": "",
            "model": "",
            "include[]": "",
            "known_speaker_names[]": "",
            "temperature": "",
            "response_format": "diarized_json",
            "language": "",
            "chunking_strategy": "",
        },
    )

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_transcription_endpoint_returns_diarized_json():
    """diarized_json returns speaker-annotated OpenAI segments."""
    response = await _post(
        _app(), {"model": "gpt-4o-transcribe-diarize", "response_format": "diarized_json"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["segments"][0]["type"] == "transcript.text.segment"
    assert body["segments"][0]["speaker"] == "0"


class _AutoLIDSpanishASR:
    """An ASR Adapter with auto-LID that detects Spanish and hears "hola mundo"."""

    honours_prompt = False

    async def detect_language(self, pcm: bytes) -> str | None:
        return "es"

    async def transcribe_pcm(self, pcm: bytes, *, language=None, prompt=None):
        return [TranscriptToken(start=0.0, end=1.0, text=" hola mundo", probability=0.9)]


@pytest.mark.asyncio
async def test_verbose_json_reports_the_auto_detected_language_for_an_undeclared_request():
    """An undeclared-language Spanish clip reports 'language': 'es' via auto-LID.

    Exercises the real chain end to end: FullMemoryPipeline -> ASRWindowing's
    sticky LanguageState -> TranscriptionResult.detected_language ->
    MemoryTranscriptSource -> render_for_format's verbose_json body.
    """
    app = make_app(FullMemoryPipeline(asr=_AutoLIDSpanishASR(), diarization=None))

    response = await _post(app, {"model": "whisper-1", "response_format": "verbose_json"})

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "es"
    assert body["text"] == "hola mundo"
