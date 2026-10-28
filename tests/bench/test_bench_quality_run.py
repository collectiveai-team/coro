"""Quality Benchmark run against a stub transcription server on loopback."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
from support.network import LOOPBACK

from coro.bench.models.quality import ScoreError, ScoreResult

pytestmark = LOOPBACK

CANNED_DIARIZED_JSON = {
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

CANNED_HEALTH = {"status": "ok", "ready": True}


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps(CANNED_HEALTH).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
            body = json.dumps(CANNED_DIARIZED_JSON).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        pass


@pytest.fixture()
def e2e_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


def _audio(tmp_path: Path, stem: str) -> Path:
    """A placeholder WAV upload; the stub server never decodes it."""
    path = tmp_path / f"{stem}.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 200)
    return path


def _ref_stm(tmp_path: Path, stem: str, line: str) -> Path:
    path = tmp_path / f"{stem}.ref.stm"
    path.write_text(line)
    return path


def _out_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    return out_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_quality(items: list[dict], base_url: str, out_dir: Path) -> Path:
    """Run the quality subcommand and return its ``quality`` directory."""
    from coro.bench.orchestrate import run_workload

    run_workload(
        items=items,
        base_url=base_url,
        out_dir=out_dir,
        reps=1,
        subcommand="quality",
        der_collar=0.0,
        der_regions="all",
    )
    return out_dir / "quality"


class _FailAfterFirstItem:
    """Scores the first item for real and returns an error result for the rest,
    mimicking ``score_item``'s own error-result contract."""

    def __init__(self, real_score_item) -> None:
        self._real_score_item = real_score_item
        self.calls = 0

    def __call__(self, ref_path, hyp_path, **kwargs):
        self.calls += 1
        if self.calls > 1:
            return ScoreResult(
                metrics=None,
                error=ScoreError(type="RuntimeError", message="scoring failed for meeting2"),
            )
        return self._real_score_item(ref_path, hyp_path, **kwargs)


@pytest.fixture()
def one_item_quality_dir(e2e_server, tmp_path: Path) -> Path:
    items = [
        {
            "item_id": "meeting1",
            "audio_path": _audio(tmp_path, "meeting1"),
            "ref_stm_path": _ref_stm(
                tmp_path, "meeting1", "meeting1 1 SPEAKER_00 0.000 1.500 hello world\n"
            ),
            "audio_seconds": 3.5,
        }
    ]
    return _run_quality(items, e2e_server, _out_dir(tmp_path))


class TestQualityRun:
    def test_quality_run_writes_the_item_score(self, one_item_quality_dir: Path):
        assert (one_item_quality_dir / "meeting1.json").exists()

        item_data = _read_json(one_item_quality_dir / "meeting1.json")
        assert item_data["session_id"] == "meeting1"
        assert item_data["audio_seconds"] == pytest.approx(3.5, abs=1e-9)
        assert item_data["metrics"] is not None
        assert "cpwer" in item_data["metrics"]
        assert "der" in item_data["metrics"]

    def test_quality_run_writes_the_summary(self, one_item_quality_dir: Path):
        assert (one_item_quality_dir / "summary.json").exists()

        summary = _read_json(one_item_quality_dir / "summary.json")
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert "per_item" in summary
        assert summary["per_item"][0]["session_id"] == "meeting1"

    def test_quality_run_isolates_failures(self, e2e_server, tmp_path: Path):
        from coro.bench import quality as quality_mod

        items = [
            {
                "item_id": stem,
                "audio_path": _audio(tmp_path, stem),
                "ref_stm_path": _ref_stm(tmp_path, stem, f"{stem} 1 A 0.0 1.0 {word}\n"),
                "audio_seconds": 1.0,
            }
            for stem, word in (("meeting1", "hello"), ("meeting2", "world"))
        ]
        scorer = _FailAfterFirstItem(quality_mod.score_item)

        with patch.object(quality_mod, "score_item", autospec=True, side_effect=scorer):
            quality_dir = _run_quality(items, e2e_server, _out_dir(tmp_path))

        item2 = _read_json(quality_dir / "meeting2.json")
        assert item2["metrics"] is None
        assert "error" in item2

        summary = _read_json(quality_dir / "summary.json")
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 1
