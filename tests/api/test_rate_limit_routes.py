"""Every transcription entry point rejects an over-limit client with 429 (ADR 0024).

REST and SSE share the OpenAI route, Deepgram has its own REST route and the
live WebSocket; each renders the rejection in its vendor's shape with a
``Retry-After`` hint, and none of them hands a rejected request to the pipeline.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from coro.core.models import TranscriptDoneEvent
from coro.settings import ServerSettings
from support.factories import FakePipeline, make_app, make_wav

ONE_SECOND_WAV = make_wav(frames=16000)
ONE_SECOND_PCM = b"\x00\x00" * 16000


class _CountingPipeline(FakePipeline):
    """Counts transcriptions, so a test can show a rejection spent no CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def transcribe(self, audio, *, language=None, prompt=None):
        self.calls += 1
        return await super().transcribe(audio, language=language, prompt=prompt)

    async def stream(self, audio, *, language=None, prompt=None):
        self.calls += 1
        yield TranscriptDoneEvent(text=json.dumps({"segments": [], "word_segments": []}))


class _FakeASR:
    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        return []


def _client(pipeline=None, **limits) -> TestClient:
    settings = ServerSettings(_env_file=None, **limits)
    app = make_app(pipeline or _CountingPipeline(), settings=settings)
    app.state.runtime.asr_adapter = _FakeASR()
    return TestClient(app)


def _denied_upgrade(client: TestClient) -> WebSocketDenialResponse:
    """Open a WebSocket the server must refuse, returning the HTTP denial."""
    with pytest.raises(WebSocketDenialResponse) as denied:
        client.websocket_connect("/v1/listen").__enter__()
    return denied.value


def _openai(client: TestClient, *, stream: bool = False, wav: bytes = ONE_SECOND_WAV):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "x", "stream": "true" if stream else "false"},
    )


def _deepgram(client: TestClient, wav: bytes = ONE_SECOND_WAV):
    return client.post("/v1/listen", content=wav, headers={"content-type": "audio/wav"})


class TestRequestsPerMinute:
    def test_openai_rest_rejects_beyond_the_burst_with_retry_after(self):
        pipeline = _CountingPipeline()
        client = _client(pipeline, rate_limit_requests_per_minute=2)
        statuses = [_openai(client).status_code for _ in range(3)]
        rejected = _openai(client)
        assert statuses == [200, 200, 429]
        assert rejected.json()["error"]["code"] == "rate_limit_exceeded"
        assert int(rejected.headers["Retry-After"]) >= 1
        assert pipeline.calls == 2

    def test_openai_sse_is_rejected_before_the_stream_starts(self):
        pipeline = _CountingPipeline()
        client = _client(pipeline, rate_limit_requests_per_minute=1)
        first = _openai(client, stream=True)
        rejected = _openai(client, stream=True)
        assert first.status_code == 200
        assert rejected.status_code == 429
        assert rejected.headers["content-type"].startswith("application/json")
        assert pipeline.calls == 1

    def test_deepgram_rest_rejects_in_deepgram_shape(self):
        client = _client(rate_limit_requests_per_minute=1)
        assert _deepgram(client).status_code == 200
        rejected = _deepgram(client)
        assert rejected.status_code == 429
        assert rejected.json()["err_code"] == "TOO_MANY_REQUESTS"
        assert int(rejected.headers["Retry-After"]) >= 1

    def test_websocket_upgrade_is_denied_with_429(self):
        client = _client(rate_limit_requests_per_minute=1)
        with client.websocket_connect("/v1/listen") as ws:
            ws.send_text(json.dumps({"type": "CloseStream"}))
            while json.loads(ws.receive_text())["type"] != "Metadata":
                pass
        denied = _denied_upgrade(client)
        assert denied.status_code == 429
        assert int(denied.headers["Retry-After"]) >= 1

    def test_all_entry_points_share_one_budget_per_client(self):
        client = _client(rate_limit_requests_per_minute=2)
        assert _openai(client).status_code == 200
        assert _deepgram(client).status_code == 200
        assert _openai(client, stream=True).status_code == 429

    def test_zero_disables_the_limit(self):
        client = _client(rate_limit_requests_per_minute=0)
        assert {_openai(client).status_code for _ in range(100)} == {200}


# 0.05 min/h is a 3 s budget; requests are left unlimited.
AUDIO_LIMIT = {"rate_limit_requests_per_minute": 0, "rate_limit_audio_minutes_per_hour": 0.05}


class TestAudioMinutesPerHour:
    def test_an_upload_that_does_not_fit_is_rejected_before_transcription(self):
        pipeline = _CountingPipeline()
        client = _client(pipeline, **AUDIO_LIMIT)
        two_seconds = make_wav(frames=32000)
        assert _openai(client, wav=two_seconds).status_code == 200
        rejected = _openai(client, wav=two_seconds)
        assert rejected.status_code == 429
        assert int(rejected.headers["Retry-After"]) >= 1
        assert pipeline.calls == 1

    def test_deepgram_rest_charges_the_same_budget(self):
        client = _client(**AUDIO_LIMIT)
        assert _deepgram(client, make_wav(frames=32000)).status_code == 200
        assert _deepgram(client, make_wav(frames=32000)).status_code == 429

    def test_a_stream_that_runs_out_mid_way_gets_an_error_and_1008(self):
        client = _client(**AUDIO_LIMIT)
        with client.websocket_connect("/v1/listen?encoding=linear16&sample_rate=16000") as ws:
            for _ in range(5):
                ws.send_bytes(ONE_SECOND_PCM)
            frame = json.loads(ws.receive_text())
            closed = ws.receive()
        assert frame["type"] == "Error"
        assert frame["description"] == "Rate limit exceeded"
        assert closed == {"type": "websocket.close", "code": 1008, "reason": ""}

    def test_a_stream_with_no_audio_left_is_denied_at_connect(self):
        client = _client(**AUDIO_LIMIT)
        assert _openai(client, wav=make_wav(frames=16000 * 5)).status_code == 200
        denied = _denied_upgrade(client)
        assert denied.status_code == 429
