"""The benchmark can drive the Deepgram-native ``POST /v1/listen`` endpoint.

This is the only wire surface coro serves that carries per-word speaker
labels. ``diarized_json`` has no word field, so a run over the OpenAI endpoint
can only ever score the segment-level speaker summary.

Deepgram's request contract differs from OpenAI's in three ways that all have
to be right or the request is rejected or silently unattributed: a raw body
rather than multipart, an audio content type rather than JSON, and explicit
``diarize``/``utterances`` flags.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from coro.bench.errors import ServerUnreachableError
from coro.bench.transport import transcribe_audio_deepgram

AUDIO_BYTES = b"RIFF" + b"\x00" * 100

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


class _ListenStubHandler(BaseHTTPRequestHandler):
    captured_path: str = ""
    captured_body: bytes = b""
    captured_content_type: str = ""

    def do_POST(self):
        if self.path.split("?")[0] == "/v1/listen":
            content_length = int(self.headers.get("Content-Length", 0))
            _ListenStubHandler.captured_path = self.path
            _ListenStubHandler.captured_body = self.rfile.read(content_length)
            _ListenStubHandler.captured_content_type = self.headers.get("Content-Type", "")
            resp = json.dumps(DEEPGRAM_RESPONSE).encode()
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
def listen_stub_server():
    _ListenStubHandler.captured_path = ""
    _ListenStubHandler.captured_body = b""
    _ListenStubHandler.captured_content_type = ""
    server = HTTPServer(("127.0.0.1", 0), _ListenStubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


class TestTranscribeAudioDeepgram:
    def test_posts_the_raw_audio_body_to_listen(self, listen_stub_server, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(AUDIO_BYTES)

        result = transcribe_audio_deepgram(listen_stub_server, audio)

        assert _ListenStubHandler.captured_body == AUDIO_BYTES
        assert result["results"]["channels"][0]["alternatives"][0]["words"][0]["word"] == "hello"

    def test_requests_diarize_and_utterances(self, listen_stub_server, tmp_path: Path):
        """Both default to false at /v1/listen, so an unasked request has no speakers."""
        audio = tmp_path / "test.wav"
        audio.write_bytes(AUDIO_BYTES)

        transcribe_audio_deepgram(listen_stub_server, audio)

        query = parse_qs(urlparse(_ListenStubHandler.captured_path).query)
        assert query["diarize"] == ["true"]
        assert query["utterances"] == ["true"]

    def test_does_not_send_a_json_content_type(self, listen_stub_server, tmp_path: Path):
        """/v1/listen reads application/json as Deepgram URL ingest and refuses it.

        urllib defaults an unlabelled body to ``x-www-form-urlencoded``, so the
        header is set explicitly rather than left to chance.
        """
        audio = tmp_path / "test.wav"
        audio.write_bytes(AUDIO_BYTES)

        transcribe_audio_deepgram(listen_stub_server, audio)

        content_type = _ListenStubHandler.captured_content_type.split(";")[0].strip()
        assert content_type != "application/json"
        assert content_type == "audio/x-wav"

    def test_unreachable_server_raises_server_unreachable(self, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(AUDIO_BYTES)

        with pytest.raises(ServerUnreachableError):
            transcribe_audio_deepgram("http://127.0.0.1:1", audio, timeout_seconds=0.5)
