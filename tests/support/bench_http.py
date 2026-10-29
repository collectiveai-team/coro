"""Stub coro servers on a loopback port, for the bench tests that run a workload.

Each handler answers ``/health`` and ``/v1/audio/transcriptions`` with canned
bodies, so the bench drives the same HTTP calls it sends to a real server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

from coro.bench.models.resource import ProcessTreeSample

TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"

CANNED_HEALTH = {
    "status": "ok",
    "ready": True,
    "warmup_ready": True,
    "startup_selection": {},
    "capability_readiness": {"asr": True},
}

CANNED_RESPONSE = {
    "task": "transcribe",
    "duration": 3.5,
    "text": "hello world",
    "segments": [],
    "usage": {"type": "duration", "seconds": 4},
}


class StubHandler(BaseHTTPRequestHandler):
    """Answer health with ``health`` and transcriptions with ``response``."""

    health: ClassVar[dict[str, object]] = CANNED_HEALTH
    response: ClassVar[dict[str, object]] = CANNED_RESPONSE

    def do_GET(self):
        if self.path == "/health":
            self._send(200, "application/json", json.dumps(self.health).encode())
        else:
            self._send(404)

    def do_POST(self):
        if self.path == TRANSCRIPTIONS_PATH:
            self.on_upload(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.send_transcription()
        else:
            self._send(404)

    def on_upload(self, body: bytes) -> None:
        """Receive the raw multipart body of one transcription request."""

    def send_transcription(self) -> None:
        """Answer one transcription request with the canned JSON response."""
        self._send(200, "application/json", json.dumps(self.response).encode())

    def _send(self, status: int, content_type: str | None = None, body: bytes = b"") -> None:
        self.send_response(status)
        if content_type is not None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 -- BaseHTTPRequestHandler signature
        pass


def uploaded_filename(body: bytes) -> str:
    """Pull the multipart upload's filename out of a captured request body."""
    match = re.search(rb'filename="([^"]+)"', body)
    return match.group(1).decode() if match else ""


CANNED_DIARIZED_RESPONSE = {
    "task": "transcribe",
    "duration": 3.5,
    "text": "hello world from test",
    "segments": [
        {
            "type": "transcript.text.segment",
            "id": "seg_001",
            "start": 0.0,
            "end": 1.5,
            "text": "hello world",
            "speaker": "SPEAKER_00",
        },
        {
            "type": "transcript.text.segment",
            "id": "seg_002",
            "start": 1.5,
            "end": 3.5,
            "text": "from test",
            "speaker": "SPEAKER_01",
        },
    ],
    "usage": {"type": "duration", "seconds": 4},
}

CANNED_DIARIZED_HEALTH = {
    "status": "ok",
    "ready": True,
    "warmup_ready": True,
    "startup_selection": {
        "pipeline": "full-memory",
        "asr_provider": "faster-whisper",
        "asr_model": "openai/whisper-medium",
        "diarization_provider": "none",
        "diarization_model": None,
    },
    "capability_readiness": {
        "asr": True,
        "diarization": "disabled",
        "transcription": True,
    },
}


class RecordingHandler(StubHandler):
    """Answer with diarized segments and record each uploaded filename."""

    health: ClassVar[dict[str, object]] = CANNED_DIARIZED_HEALTH
    response: ClassVar[dict[str, object]] = CANNED_DIARIZED_RESPONSE
    uploads: ClassVar[list[str]] = []
    """Filenames posted to the transcription endpoint, in request order."""

    def on_upload(self, body: bytes) -> None:
        RecordingHandler.uploads.append(uploaded_filename(body))


@contextmanager
def serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    """Serve ``handler`` on a free loopback port and yield its base URL."""
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()


def mock_sample_fn(pid: int) -> ProcessTreeSample:
    """A fixed process-tree sample, so a run needs no real server process."""
    return ProcessTreeSample(
        pids={pid, pid + 1},
        pss_kb=100000 + pid,
        uss_kb=50000,
        rss_kb=120000,
        vsz_kb=200000,
        cpu_user_s=1.5,
        cpu_system_s=0.3,
        rchar=1000000,
        wchar=500000,
        read_bytes=300000,
        write_bytes=200000,
        thread_count=8,
    )


def placeholder_audio(tmp_path: Path, stem: str, size: int = 200) -> Path:
    """A placeholder WAV upload; the stub server never decodes it."""
    path = tmp_path / f"{stem}.wav"
    path.write_bytes(b"RIFF" + b"\x00" * size)
    return path


def results_dir(tmp_path: Path) -> Path:
    """Create and return the run output directory under ``tmp_path``."""
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    return out_dir
