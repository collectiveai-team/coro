"""End-to-end test: stub server + --audio produces expected artifacts."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from coro.bench.models.quality import QualitySummary, ScoreMetrics, ScoreResult

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

CANNED_HEALTH = {
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


def _uploaded_filename(body: bytes) -> str:
    """Pull the multipart upload's filename out of a captured request body."""
    match = re.search(rb'filename="([^"]+)"', body)
    return match.group(1).decode() if match else ""


class _E2EHandler(BaseHTTPRequestHandler):
    uploads: ClassVar[list[str]] = []
    """Filenames posted to the transcription endpoint, in request order."""

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
            _E2EHandler.uploads.append(_uploaded_filename(self.rfile.read(content_length)))
            body = json.dumps(CANNED_DIARIZED_JSON).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def e2e_server():
    _E2EHandler.uploads = []
    server = HTTPServer(("127.0.0.1", 0), _E2EHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


class TestE2EAdhocAudio:
    def test_single_audio_produces_artifacts(self, e2e_server, tmp_path: Path):
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
            }
        ]

        run_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=1,
            subcommand="all",
        )

        resp_dir = out_dir / "responses"
        hyp_dir = out_dir / "hyp"
        ref_dir = out_dir / "ref"

        assert (resp_dir / "meeting1_rep1.json").exists()
        assert (hyp_dir / "meeting1.hyp.stm").exists()
        assert (ref_dir / "meeting1.ref.stm").exists()
        assert (out_dir / "manifest.json").exists()

        resp_data = json.loads((resp_dir / "meeting1_rep1.json").read_text())
        assert resp_data["task"] == "transcribe"

        hyp_text = (hyp_dir / "meeting1.hyp.stm").read_text()
        assert "meeting1" in hyp_text
        assert "SPEAKER_00" in hyp_text

        ref_text = (ref_dir / "meeting1.ref.stm").read_text()
        assert "hello world" in ref_text

    def test_manifest_contains_expected_fields(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        ref_stm = tmp_path / "meeting1.ref.stm"
        ref_stm.write_text("meeting1 1 SPK_00 0.0 1.0 test\n")

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio,
                "ref_stm_path": ref_stm,
            }
        ]

        run_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=1,
            subcommand="all",
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert "timestamp" in manifest
        assert "hostname" in manifest
        assert "cli_args" in manifest
        assert "workload_set" in manifest
        assert "server_health" in manifest
        assert manifest["workload_set"][0]["item_id"] == "meeting1"
        assert manifest["server_health"]["ready"] is True

    def test_multiple_reps_only_one_hyp_ref(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        ref_stm = tmp_path / "meeting1.ref.stm"
        ref_stm.write_text("meeting1 1 SPK_00 0.0 1.0 test\n")

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio,
                "ref_stm_path": ref_stm,
            }
        ]

        run_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=3,
            subcommand="all",
        )

        assert (out_dir / "responses" / "meeting1_rep1.json").exists()
        assert (out_dir / "responses" / "meeting1_rep2.json").exists()
        assert (out_dir / "responses" / "meeting1_rep3.json").exists()
        assert (out_dir / "hyp" / "meeting1.hyp.stm").exists()
        assert (out_dir / "ref" / "meeting1.ref.stm").exists()
        assert not (out_dir / "hyp" / "meeting1_rep2.hyp.stm").exists()

    def test_audio_without_ref_allowed_for_performance(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio,
                "ref_stm_path": None,
            }
        ]

        run_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=1,
            subcommand="performance",
        )

        assert (out_dir / "responses" / "meeting1_rep1.json").exists()
        assert not (out_dir / "ref" / "meeting1.ref.stm").exists()
        assert not (out_dir / "hyp" / "meeting1.hyp.stm").exists()

    def test_cli_audio_with_reference_stm(self, e2e_server, tmp_path: Path):
        from coro.bench.cli import parse_args

        audio = tmp_path / "test.wav"
        audio.touch()
        ref_stm = tmp_path / "test.stm"
        ref_stm.touch()

        args = parse_args(
            [
                "all",
                "--audio",
                str(audio),
                "--reference-stm",
                str(ref_stm),
                "--server-url",
                e2e_server,
            ]
        )
        assert args.audio == audio
        assert args.reference_stm == ref_stm
        assert args.server_url == e2e_server

    def test_cli_audio_without_ref_quality_rejected(self, tmp_path: Path):
        from coro.bench.cli import parse_args

        audio = tmp_path / "test.wav"
        audio.touch()

        with pytest.raises(SystemExit):
            parse_args(["quality", "--audio", str(audio)])

    def test_cli_audio_without_ref_performance_allowed(self, tmp_path: Path):
        from coro.bench.cli import parse_args

        audio = tmp_path / "test.wav"
        audio.touch()

        args = parse_args(["performance", "--audio", str(audio)])
        assert args.audio == audio
        assert args.reference_stm is None

    def test_cli_audio_without_ref_all_allowed(self, tmp_path: Path):
        from coro.bench.cli import parse_args

        audio = tmp_path / "test.wav"
        audio.touch()

        args = parse_args(["all", "--audio", str(audio)])
        assert args.audio == audio
        assert args.reference_stm is None


class TestE2EAllSubcommand:
    def test_all_produces_perf_and_quality_artifacts(self, e2e_server, tmp_path: Path):
        from unittest.mock import patch

        from coro.bench.orchestrate import run_all_workload

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
            }
        ]

        mock_score = ScoreResult(metrics=ScoreMetrics(), raw={})

        with (
            patch("coro.bench.quality.score_item", return_value=mock_score),
            patch(
                "coro.bench.quality.combine_items",
                return_value=QualitySummary(workload_set=["meeting1"], n_succeeded=1, n_failed=0),
            ),
        ):
            run_all_workload(
                items=items,
                base_url=e2e_server,
                out_dir=out_dir,
                reps=1,
                server_pid=1,
            )

        assert (out_dir / "responses" / "meeting1_rep1.json").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep1.csv").exists()
        assert (out_dir / "performance" / "summary.json").exists()
        assert (out_dir / "hyp" / "meeting1.hyp.stm").exists()
        assert (out_dir / "ref" / "meeting1.ref.stm").exists()
        assert (out_dir / "manifest.json").exists()

    def test_warmup_sends_request_before_items(self, e2e_server, tmp_path: Path):
        """Asserted at the server, so it holds whichever transport is selected."""
        from coro.bench.orchestrate import run_all_workload

        audio = tmp_path / "meeting1.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "meeting1",
                "audio_path": audio,
                "ref_stm_path": None,
            }
        ]

        warmup_path = tmp_path / "warmup.wav"
        warmup_path.write_bytes(b"RIFF" + b"\x00" * 100)

        run_all_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
            warmup_audio=warmup_path,
        )

        assert _E2EHandler.uploads == [warmup_path.name, audio.name]

        assert not any("warmup" in p.name for p in (out_dir / "responses").iterdir())

    def test_quality_scored_from_rep1_only(self, e2e_server, tmp_path: Path):
        from unittest.mock import patch

        from coro.bench.orchestrate import run_all_workload

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
            }
        ]

        mock_score = ScoreResult(metrics=ScoreMetrics(), raw={})

        with (
            patch("coro.bench.quality.score_item", return_value=mock_score),
            patch(
                "coro.bench.quality.combine_items",
                return_value=QualitySummary(workload_set=["meeting1"], n_succeeded=1, n_failed=0),
            ),
        ):
            run_all_workload(
                items=items,
                base_url=e2e_server,
                out_dir=out_dir,
                reps=3,
                server_pid=1,
            )

        assert (out_dir / "responses" / "meeting1_rep1.json").exists()
        assert (out_dir / "responses" / "meeting1_rep2.json").exists()
        assert (out_dir / "responses" / "meeting1_rep3.json").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep1.csv").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep2.csv").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep3.csv").exists()
        assert (out_dir / "hyp" / "meeting1.hyp.stm").exists()
        assert not (out_dir / "hyp" / "meeting1_rep2.hyp.stm").exists()

    def test_adhoc_audio_without_ref_skips_quality(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_all_workload

        audio = tmp_path / "my.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {
                "item_id": "my",
                "audio_path": audio,
                "ref_stm_path": None,
            }
        ]

        run_all_workload(
            items=items,
            base_url=e2e_server,
            out_dir=out_dir,
            reps=1,
            server_pid=1,
        )

        assert (out_dir / "responses" / "my_rep1.json").exists()
        assert (out_dir / "performance" / "resource_my_rep1.csv").exists()
        assert (out_dir / "performance" / "summary.json").exists()
        assert not (out_dir / "hyp" / "my.hyp.stm").exists()
        assert not (out_dir / "quality" / "my.json").exists()

        quality_summary = json.loads((out_dir / "quality" / "summary.json").read_text())
        assert quality_summary["n_skipped"] == 1
        assert quality_summary["n_succeeded"] == 0

    def test_mixed_items_with_skip_and_quality(self, e2e_server, tmp_path: Path):
        from unittest.mock import patch

        from coro.bench.orchestrate import run_all_workload

        audio1 = tmp_path / "IB4001.wav"
        audio1.write_bytes(b"RIFF" + b"\x00" * 200)
        ref1 = tmp_path / "IB4001.ref.stm"
        ref1.write_text("IB4001 1 SPEAKER_00 0.000 1.500 hello world\n")

        audio2 = tmp_path / "IN1001.wav"
        audio2.write_bytes(b"RIFF" + b"\x00" * 200)
        ref2 = tmp_path / "IN1001.ref.stm"
        ref2.write_text("IN1001 1 SPEAKER_00 0.000 1.500 test data\n")

        audio3 = tmp_path / "adhoc.wav"
        audio3.write_bytes(b"RIFF" + b"\x00" * 200)

        out_dir = tmp_path / "results"
        out_dir.mkdir()

        items = [
            {"item_id": "IB4001", "audio_path": audio1, "ref_stm_path": ref1},
            {"item_id": "IN1001", "audio_path": audio2, "ref_stm_path": ref2},
            {"item_id": "adhoc", "audio_path": audio3, "ref_stm_path": None},
        ]

        mock_score = ScoreResult(metrics=ScoreMetrics(), raw={})

        with (
            patch("coro.bench.quality.score_item", return_value=mock_score),
            patch(
                "coro.bench.quality.combine_items",
                return_value=QualitySummary(
                    workload_set=["IB4001", "IN1001"], n_succeeded=2, n_failed=0
                ),
            ),
        ):
            run_all_workload(
                items=items,
                base_url=e2e_server,
                out_dir=out_dir,
                reps=2,
                server_pid=1,
            )

        assert (out_dir / "responses" / "IB4001_rep1.json").exists()
        assert (out_dir / "responses" / "IB4001_rep2.json").exists()
        assert (out_dir / "responses" / "IN1001_rep1.json").exists()
        assert (out_dir / "responses" / "adhoc_rep1.json").exists()

        assert (out_dir / "performance" / "resource_IB4001_rep1.csv").exists()
        assert (out_dir / "performance" / "resource_IB4001_rep2.csv").exists()
        assert (out_dir / "performance" / "resource_adhoc_rep1.csv").exists()
        assert (out_dir / "performance" / "summary.json").exists()

        assert (out_dir / "hyp" / "IB4001.hyp.stm").exists()
        assert (out_dir / "hyp" / "IN1001.hyp.stm").exists()
        assert not (out_dir / "hyp" / "adhoc.hyp.stm").exists()

        assert (out_dir / "quality" / "IB4001.json").exists()
        assert (out_dir / "quality" / "IN1001.json").exists()
        assert not (out_dir / "quality" / "adhoc.json").exists()

        quality_summary = json.loads((out_dir / "quality" / "summary.json").read_text())
        assert quality_summary["n_skipped"] == 1
        assert quality_summary["n_succeeded"] == 2
