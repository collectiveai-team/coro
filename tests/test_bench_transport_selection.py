"""One place decides which endpoint a benchmark run talks to.

The choice used to be an ``if stream:`` duplicated at each workload call site,
which is why the quality workload — the one that computes WDER — had no choice
at all and could only ever reach the OpenAI endpoint.

These tests assert the endpoint actually requested, not which function object
came back, so they survive a refactor of the selection internals.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from coro.bench.transport import select_transport

OPENAI_PATH = "/v1/audio/transcriptions"
LISTEN_PATH = "/v1/listen"

OPENAI_RESPONSE = {
    "text": "hello world",
    "segments": [{"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "0"}],
}

DEEPGRAM_RESPONSE = {
    "metadata": {"request_id": "abc123", "duration": 2.0, "channels": 1},
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "hello world",
                        "confidence": 0.9,
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 1.0, "speaker": 0},
                            {"word": "world", "start": 1.0, "end": 2.0, "speaker": 1},
                        ],
                    }
                ]
            }
        ]
    },
}

SSE_BODY = (
    b"event: transcript.text.delta\r\n"
    b'data: {"type": "transcript.text.delta", "delta": "hello"}\r\n'
    b"\r\n"
    b"event: transcript.text.done\r\n"
    b'data: {"type": "transcript.text.done", "text": '
    + json.dumps(json.dumps(OPENAI_RESPONSE)).encode()
    + b"}\r\n\r\n"
)


class _RecordingHandler(BaseHTTPRequestHandler):
    """Serves all three shapes and records what was asked for."""

    requests: ClassVar[list[tuple[str, str]]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        path = self.path
        _RecordingHandler.requests.append((path, self.headers.get("Content-Type", "")))

        if path.split("?")[0] == LISTEN_PATH:
            self._json(DEEPGRAM_RESPONSE)
        elif b'name="stream"' in body:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(SSE_BODY)
            self.wfile.flush()
        else:
            self._json(OPENAI_RESPONSE)

    def _json(self, payload: Any) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def recording_server():
    _RecordingHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


@pytest.fixture()
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "test.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 100)
    return path


def _paths() -> list[str]:
    return [path.split("?")[0] for path, _ in _RecordingHandler.requests]


class TestSelectTransport:
    def test_default_selects_the_openai_multipart_endpoint(self, recording_server, audio):
        """The default must not move; every existing run depends on it."""
        result, ttft = select_transport()(recording_server, audio)

        assert _paths() == [OPENAI_PATH]
        assert _RecordingHandler.requests[0][1].startswith("multipart/form-data")
        assert ttft is None
        assert result["text"] == "hello world"

    def test_stream_selects_the_sse_endpoint_and_times_first_delta(
        self, recording_server, audio
    ):
        result, ttft = select_transport(stream=True)(recording_server, audio)

        assert _paths() == [OPENAI_PATH]
        assert ttft is not None
        assert result["segments"][0]["text"] == "hello world"

    def test_deepgram_selects_the_listen_endpoint(self, recording_server, audio):
        result, ttft = select_transport(deepgram=True)(recording_server, audio)

        assert _paths() == [LISTEN_PATH]
        assert ttft is None
        assert result["results"]["channels"][0]["alternatives"][0]["words"][0]["word"] == "hello"
