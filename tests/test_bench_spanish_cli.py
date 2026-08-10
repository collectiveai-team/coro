"""CLI-level tests: the Spanish preset runs through the Quality Benchmark path."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from coro.bench.cli import main, parse_args

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
    "capability_readiness": {"asr": True, "diarization": "disabled", "transcription": True},
}


def _canned_body(text: str) -> bytes:
    """Serialise a one-segment transcription response for the stub server."""
    payload = {
        "task": "transcribe",
        "duration": 1.0,
        "text": text,
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": "seg_001",
                "start": 0.0,
                "end": 1.0,
                "text": text,
                "speaker": "SPEAKER_00",
            }
        ],
    }
    return json.dumps(payload).encode()


class _SpanishHandler(BaseHTTPRequestHandler):
    transcript = "hola cómo está el año"

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(CANNED_HEALTH).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.send_response(404)
            self.end_headers()
            return
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = _canned_body(type(self).transcript)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def spanish_server():
    server = HTTPServer(("127.0.0.1", 0), _SpanishHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join()


def _argv(server: str, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "coro-bench",
        "quality",
        "--spanish-preset",
        "fleurs",
        "--spanish-root",
        str(tmp_path / "corpora"),
        "--server-url",
        server,
        "--out-dir",
        str(tmp_path / "run"),
        *extra,
    ]


class TestSpanishCliArgs:
    def test_preset_choices_come_from_the_registry(self):
        args = parse_args(["quality", "--spanish-preset", "calibration"])

        assert args.spanish_preset == "calibration"

    def test_preset_and_clips_dir_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            parse_args(["quality", "--spanish-preset", "fleurs", "--clips-dir", "clips"])

    def test_fetch_plan_requires_a_preset(self):
        with pytest.raises(SystemExit):
            parse_args(["quality", "--spanish-fetch-plan"])

    def test_quarantined_reference_stm_is_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["quality", "--audio", "a.wav", "--reference-stm", "run/hyp/IB4001.hyp.stm"])

    def test_calibration_margin_defaults_are_exposed(self):
        from coro.bench.calibration import DEFAULT_CALIBRATION_MARGIN

        args = parse_args(["quality", "--spanish-preset", "mls"])

        assert args.calibration_margin == DEFAULT_CALIBRATION_MARGIN


class TestSpanishQualityRun:
    def test_preset_scores_through_the_quality_benchmark_path(
        self, spanish_server, tmp_path: Path, fake_spanish_corpus, monkeypatch
    ):
        monkeypatch.setattr(
            "sys.argv", _argv(spanish_server, tmp_path, "--calibration-margin", "1")
        )

        main()

        run = tmp_path / "run"
        assert (run / "quality" / "fleurs-101.json").exists()
        assert (run / "quality" / "summary.json").exists()
        assert (run / "hyp" / "fleurs-101.hyp.stm").exists()
        assert (run / "ref" / "fleurs-101.ref.stm").exists()
        assert (run / "REPORT.md").exists()

        summary = json.loads((run / "quality" / "summary.json").read_text())
        assert summary["n_succeeded"] == 2

    def test_calibration_artifact_records_the_published_figure(
        self, spanish_server, tmp_path: Path, fake_spanish_corpus, monkeypatch
    ):
        monkeypatch.setattr(
            "sys.argv", _argv(spanish_server, tmp_path, "--calibration-margin", "1")
        )

        main()

        report = json.loads((tmp_path / "run" / "quality" / "calibration.json").read_text())
        assert report["model_id"] == "openai/whisper-medium"
        assert report["metric"] == "normalized_orcwer"
        assert report["outcomes"][0]["corpus"] == "fleurs"
        assert report["outcomes"][0]["published_wer"] == pytest.approx(0.036, abs=1e-12)

    def test_deviation_beyond_the_margin_fails_the_run(
        self, spanish_server, tmp_path: Path, fake_spanish_corpus, monkeypatch
    ):
        monkeypatch.setattr(_SpanishHandler, "transcript", "palabras completamente distintas aqui")
        monkeypatch.setattr("sys.argv", _argv(spanish_server, tmp_path))

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 3
        report = json.loads((tmp_path / "run" / "quality" / "calibration.json").read_text())
        assert report["failed"] is True

    def test_no_calibration_reports_without_failing(
        self, spanish_server, tmp_path: Path, fake_spanish_corpus, monkeypatch
    ):
        monkeypatch.setattr(_SpanishHandler, "transcript", "palabras completamente distintas aqui")
        monkeypatch.setattr("sys.argv", _argv(spanish_server, tmp_path, "--no-calibration"))

        main()

        report = json.loads((tmp_path / "run" / "quality" / "calibration.json").read_text())
        assert report["failed"] is True
