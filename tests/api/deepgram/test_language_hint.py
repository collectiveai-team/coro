"""Language preferences are transport-compatible, not model instructions."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from coro.api.asyncapi import build_asyncapi_document
from coro.audio import SAMPLE_RATE
from coro.core.models import TranscriptToken
from coro.settings import ServerSettings
from support.adapters import RecordingASR
from support.factories import FakePipeline, make_app, make_wav


class RecordingPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.languages: list[str | None] = []

    async def transcribe(self, audio, *, language=None, prompt=None):
        self.languages.append(language)
        return self.result


@pytest.mark.parametrize(
    "hint", ["language_hint=es", "language_hint=es&language_hint=en", "language_hint="]
)
@pytest.mark.parametrize("language,expected", [("", None), ("&language=ES_ar", "es-AR")])
async def test_rest_hints_leave_language_and_results_unchanged(hint, language, expected):
    pipeline = RecordingPipeline()
    app = make_app(pipeline, settings=ServerSettings(_env_file=None))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        responses = [
            await client.post(
                f"/v1/listen?{query}",
                content=make_wav(),
                headers={"Content-Type": "audio/wav"},
            )
            for query in (language.lstrip("&"), f"{hint}{language}")
        ]
    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].json()["results"] == responses[1].json()["results"]
    assert pipeline.languages == [expected, expected]


@pytest.mark.parametrize(
    "hint", ["language_hint=es", "language_hint=es&language_hint=en", "language_hint="]
)
@pytest.mark.parametrize("language,expected", [("", None), ("&language=ES_ar", "es-AR")])
def test_websocket_hints_do_not_select_or_override_language(hint, language, expected):
    asr = RecordingASR([TranscriptToken(start=0.0, end=0.5, text=" hola")])
    app = make_app(FakePipeline(), settings=ServerSettings(_env_file=None))
    app.state.runtime.asr_adapter = asr
    with TestClient(app).websocket_connect(f"/v1/listen?{hint}{language}") as ws:
        ws.send_bytes(b"\x00\x00" * SAMPLE_RATE)
        ws.send_json({"type": "CloseStream"})
        frames = []
        while True:
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            if frame["type"] == "Metadata":
                break
    assert frames[0]["type"] == "Results"
    assert frames[0]["channel"]["alternatives"][0]["transcript"] == "hola"
    assert asr.languages == [expected]


def test_contracts_describe_language_preferences_as_ignored():
    client = TestClient(make_app(FakePipeline(), settings=ServerSettings(_env_file=None)))
    parameters = client.get("/openapi.json").json()["paths"]["/v1/listen"]["post"]["parameters"]
    hint = next(parameter for parameter in parameters if parameter["name"] == "language_hint")
    assert "Accepted but ignored" in hint["description"]
    document = json.loads(build_asyncapi_document().to_json())
    query = document["channels"]["deepgramLiveStream"]["bindings"]["ws"]["query"]
    assert query["properties"]["language_hint"]["type"] == "array"
    assert "Accepted but ignored" in query["properties"]["language_hint"]["description"]
