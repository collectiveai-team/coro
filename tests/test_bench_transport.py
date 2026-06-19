"""Tests for HTTP transport layer."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from coro.bench.transport import transcribe_audio, transcribe_audio_sse

CANNED_RESPONSE = {
    "task": "transcribe",
    "duration": 5.0,
    "text": "hello world",
    "segments": [
        {
            "type": "transcript.text.segment",
            "id": "seg_001",
            "start": 0.0,
            "end": 2.5,
            "text": "hello",
            "speaker": "SPEAKER_00",
        },
        {
            "type": "transcript.text.segment",
            "id": "seg_002",
            "start": 2.5,
            "end": 5.0,
            "text": "world",
            "speaker": "SPEAKER_01",
        },
    ],
    "usage": {"type": "duration", "seconds": 5},
}


class _StubHandler(BaseHTTPRequestHandler):
    captured_body: bytes = b""

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            _StubHandler.captured_body = body
            resp = json.dumps(CANNED_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def stub_server():
    _StubHandler.captured_body = b""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


class TestTranscribeAudio:
    def test_sends_post_and_returns_json(self, stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        result = transcribe_audio(stub_server, audio)
        assert result["task"] == "transcribe"
        assert result["text"] == "hello world"
        assert len(result["segments"]) == 2

    def test_sends_diarized_json_format(self, stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        transcribe_audio(stub_server, audio)

        assert b"diarized_json" in _StubHandler.captured_body

    def test_timeout_raises_on_unreachable(self, tmp_path: Path):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(0)

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        with pytest.raises((TimeoutError, OSError)):
            transcribe_audio(
                f"http://127.0.0.1:{port}",
                audio,
                timeout_seconds=0.5,
            )
        s.close()


SSE_SEGMENTS = [{"start": 0.0, "end": 1.5, "text": "hello world", "speaker": "A"}]

SSE_RESPONSE_BODY = (
    b"event: transcript.text.delta\r\n"
    b'data: {"type": "transcript.text.delta", "delta": "hello"}\r\n'
    b"\r\n"
    b"event: transcript.text.delta\r\n"
    b'data: {"type": "transcript.text.delta", "delta": " world"}\r\n'
    b"\r\n"
    b"event: transcript.text.done\r\n"
    + b'data: {"type": "transcript.text.done", "text": '
    + json.dumps(json.dumps({"segments": SSE_SEGMENTS})).encode()
    + b"}\r\n"
    b"\r\n"
)


class _SSEStubHandler(BaseHTTPRequestHandler):
    captured_body: bytes = b""

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            _SSEStubHandler.captured_body = body
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(SSE_RESPONSE_BODY)
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def sse_stub_server():
    _SSEStubHandler.captured_body = b""
    server = HTTPServer(("127.0.0.1", 0), _SSEStubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


class TestTranscribeAudioSSE:
    def test_sends_stream_true_form_field(self, sse_stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        transcribe_audio_sse(sse_stub_server, audio)

        assert b"stream" in _SSEStubHandler.captured_body

    def test_records_time_to_first_delta(self, sse_stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        _, ttft = transcribe_audio_sse(sse_stub_server, audio)

        assert ttft is not None
        assert ttft >= 0.0
        assert ttft < 5.0

    def test_extracts_diarized_json_from_done_event(self, sse_stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        result, _ = transcribe_audio_sse(sse_stub_server, audio)

        assert "segments" in result
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "hello world"
        assert result["segments"][0]["speaker"] == "A"

    def test_returns_tuple_of_dict_and_float(self, sse_stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        result = transcribe_audio_sse(sse_stub_server, audio)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], float)
