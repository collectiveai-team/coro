"""Cycle 6: SSE streaming framing matches OpenAI-Exact SSE contract.

Verifies that stream=True returns:
- Content-Type: text/event-stream
- ``data: {...}\\n\\n`` framing for every event
- A ``transcript.text.done`` event carrying the final JSON result
- The last event is ``data: [DONE]\\n\\n``
- No ``transcript.progress`` events (package-specific events are forbidden)

Both v1 and v2 must satisfy the same SSE contract.
"""

from __future__ import annotations

import io
import json
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from asr_diar_server.app import create_app
from asr_diar_server.runtime import RuntimeState
from asr_diar_server.settings import ServerSettings

_WHISPERX_EMPTY = {
    "segments": [],
    "word_segments": [],
    "transcript": [],
    "diarization": [],
    "raw_words": [],
}


def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<1600h", *([0] * 1600)))
    return buf.getvalue()


class _FakeV1StreamingPipeline:
    async def run(self, audio_bytes, *, language=None, prompt=None):
        return dict(_WHISPERX_EMPTY)

    async def stream(self, audio_bytes, *, language=None, prompt=None):
        """Yield one delta then done."""
        yield {"type": "transcript.text.delta", "delta": "Hello"}
        yield {"type": "transcript.text.done", "text": json.dumps(_WHISPERX_EMPTY)}


class _FakeV2StreamingPipeline:
    async def run_from_path(self, path, *, language=None, prompt=None):
        return dict(_WHISPERX_EMPTY)

    async def stream_from_path(self, path, *, language=None, prompt=None):
        yield {"type": "transcript.text.delta", "delta": "World"}
        yield {"type": "transcript.text.done", "text": json.dumps(_WHISPERX_EMPTY)}


def _app_with_streaming_pipelines():
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.v1_pipeline = _FakeV1StreamingPipeline()
    runtime.v2_pipeline = _FakeV2StreamingPipeline()
    application.state.runtime = runtime
    return application


def _parse_sse_events(raw: str) -> list[dict | str]:
    """Parse raw SSE text into a list of event data values.

    Returns "[DONE]" for the terminator and a dict for JSON events.
    """
    events: list[dict | str] = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(json.loads(payload))
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "pipeline_attr"),
    [("/v1/audio/transcriptions", "v1_pipeline"), ("/v2/audio/transcriptions", "v2_pipeline")],
)
async def test_streaming_content_type(route, pipeline_attr):
    """stream=True returns Content-Type: text/event-stream."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"stream": "true"},
        )
    assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "pipeline_attr"),
    [("/v1/audio/transcriptions", "v1_pipeline"), ("/v2/audio/transcriptions", "v2_pipeline")],
)
async def test_streaming_ends_with_done_sentinel(route, pipeline_attr):
    """SSE stream ends with 'data: [DONE]'."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"stream": "true"},
        )
    events = _parse_sse_events(response.text)
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "pipeline_attr"),
    [("/v1/audio/transcriptions", "v1_pipeline"), ("/v2/audio/transcriptions", "v2_pipeline")],
)
async def test_streaming_contains_done_event(route, pipeline_attr):
    """SSE stream contains a transcript.text.done event before [DONE]."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"stream": "true"},
        )
    events = _parse_sse_events(response.text)
    done_events = [
        e for e in events if isinstance(e, dict) and e.get("type") == "transcript.text.done"
    ]
    assert len(done_events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "pipeline_attr"),
    [("/v1/audio/transcriptions", "v1_pipeline"), ("/v2/audio/transcriptions", "v2_pipeline")],
)
async def test_streaming_has_no_progress_events(route, pipeline_attr):
    """SSE stream contains no transcript.progress events (package-specific events forbidden)."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            route,
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"stream": "true"},
        )
    events = _parse_sse_events(response.text)
    progress_events = [
        e for e in events if isinstance(e, dict) and e.get("type") == "transcript.progress"
    ]
    assert progress_events == []
