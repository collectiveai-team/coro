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


class TestQualityRun:
    def test_quality_run_produces_artifacts(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        ref_stm = tmp_path / "meeting1.ref.stm"
        ref_stm.write_text("meeting1 1 SPEAKER_00 0.000 1.500 hello world\n")

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio,
                "ref_stm_path": ref_stm,
                "audio_seconds": 3.5,
            }
        ]

        run_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=1,
            subcommand="quality",
            der_collar=0.0,
            der_regions="all",
        )

        quality_dir = out_dir / "quality"
        assert (quality_dir / "meeting1.json").exists()
        assert (quality_dir / "summary.json").exists()

        item_data = json.loads((quality_dir / "meeting1.json").read_text())
        assert item_data["session_id"] == "meeting1"
        assert item_data["audio_seconds"] == pytest.approx(3.5, abs=1e-9)
        assert item_data["metrics"] is not None
        assert "cpwer" in item_data["metrics"]
        assert "der" in item_data["metrics"]

        summary = json.loads((quality_dir / "summary.json").read_text())
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert "per_item" in summary
        assert summary["per_item"][0]["session_id"] == "meeting1"

    def test_quality_run_isolates_failures(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio1 = tmp_path / "meeting1.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 200)
        audio2 = tmp_path / "meeting2.wav"
        audio2.write_bytes(b"RIFF" + b"\x00" * 200)

        ref1 = tmp_path / "meeting1.ref.stm"
        ref1.write_text("meeting1 1 A 0.0 1.0 hello\n")
        ref2 = tmp_path / "meeting2.ref.stm"
        ref2.write_text("meeting2 1 A 0.0 1.0 world\n")

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio1,
                "ref_stm_path": ref1,
                "audio_seconds": 1.0,
            },
            {
                "item_id": "meeting2",
                "audio_path": audio2,
                "ref_stm_path": ref2,
                "audio_seconds": 1.0,
            },
        ]

        # Force the second item's scoring to fail while the first succeeds,
        # mimicking score_item's own error-result contract.
        from coro.bench import quality as quality_mod

        real_score_item = quality_mod.score_item
        call_count = [0]

        def fail_on_second(ref_path, hyp_path, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return ScoreResult(
                    metrics=None,
                    error=ScoreError(type="RuntimeError", message="scoring failed for meeting2"),
                )
            return real_score_item(ref_path, hyp_path, **kwargs)

        with patch.object(quality_mod, "score_item", autospec=True, side_effect=fail_on_second):
            run_workload(
                items=items,
                base_url=e2e_server,
                out_dir=out_dir,
                reps=1,
                subcommand="quality",
                der_collar=0.0,
                der_regions="all",
            )

        quality_dir = out_dir / "quality"
        item2 = json.loads((quality_dir / "meeting2.json").read_text())
        assert item2["metrics"] is None
        assert "error" in item2

        summary = json.loads((quality_dir / "summary.json").read_text())
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 1
