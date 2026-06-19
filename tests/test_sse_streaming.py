"""Cycle 6: SSE streaming framing matches OpenAI-Exact SSE contract.

Verifies that stream=True returns:
- Content-Type: text/event-stream
- ``data: {...}\\n\\n`` framing for every event
- A ``transcript.text.done`` event carrying the final JSON result
- The last event is ``data: [DONE]\\n\\n``
- No ``transcript.progress`` events (package-specific events are forbidden)

    The supported v1 endpoint must satisfy the SSE contract.
"""

from __future__ import annotations

import io
import json
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from coro.app import create_app
from coro.core.models import TranscriptDeltaEvent, TranscriptDoneEvent, TranscriptionResult
from coro.runtime import RuntimeState
from coro.settings import ServerSettings

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
    async def transcribe(self, audio, *, language=None, prompt=None):
        return TranscriptionResult()

    async def stream(self, audio, *, language=None, prompt=None):
        """Yield one delta then done."""
        yield TranscriptDeltaEvent(delta="Hello")
        yield TranscriptDoneEvent(text=json.dumps(_WHISPERX_EMPTY))


def _app_with_streaming_pipelines():
    from fastapi import FastAPI

    application: FastAPI = create_app(ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.pipeline = _FakeV1StreamingPipeline()
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
async def test_streaming_content_type():
    """stream=True returns Content-Type: text/event-stream."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"model": "whisper-1", "stream": "true"},
        )
    assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_streaming_ends_with_done_sentinel():
    """SSE stream ends with 'data: [DONE]'."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"model": "whisper-1", "stream": "true"},
        )
    events = _parse_sse_events(response.text)
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_streaming_contains_done_event():
    """SSE stream contains a transcript.text.done event before [DONE]."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"model": "whisper-1", "stream": "true"},
        )
    events = _parse_sse_events(response.text)
    done_events = [
        e for e in events if isinstance(e, dict) and e.get("type") == "transcript.text.done"
    ]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_streaming_has_no_progress_events():
    """SSE stream contains no transcript.progress events (package-specific events forbidden)."""
    app = _app_with_streaming_pipelines()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", _minimal_wav(), "audio/wav")},
            data={"model": "whisper-1", "stream": "true"},
        )
    events = _parse_sse_events(response.text)
    progress_events = [
        e for e in events if isinstance(e, dict) and e.get("type") == "transcript.progress"
    ]
    assert progress_events == []
