"""A benchmark run can be configured to score per-word speakers end to end.

The transport and the STM adapter are only useful together and only if a run
can actually select them. These tests drive the real workload functions against
a fake server and assert on the hypothesis STM they leave on disk — the
artefact scoring consumes — rather than on any intermediate call.

The quality workload matters most here: it is the one that computes WDER, and
it is the one that had no transport choice at all.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

OPENAI_PATH = "/v1/audio/transcriptions"
LISTEN_PATH = "/v1/listen"

# One segment spanning three speakers: the segment carries a majority summary,
# the words carry the truth, including one abstention.
OPENAI_RESPONSE = {
    "text": "hola mundo si claro",
    "segments": [{"start": 0.0, "end": 2.0, "text": "hola mundo si claro", "speaker": "1"}],
}

DEEPGRAM_RESPONSE = {
    "metadata": {"request_id": "abc123", "duration": 2.0, "channels": 1},
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "hola mundo si claro",
                        "confidence": 0.8,
                        "words": [
                            {"word": "hola", "start": 0.0, "end": 0.5, "speaker": 1},
                            {"word": "mundo", "start": 0.5, "end": 1.0, "speaker": 1},
                            {"word": "si", "start": 1.2, "end": 1.6, "speaker": 2},
                            {"word": "claro", "start": 1.6, "end": 2.0},
                        ],
                    }
                ]
            }
        ]
    },
}


class _ArmHandler(BaseHTTPRequestHandler):
    paths: ClassVar[list[str]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        path = self.path.split("?")[0]
        _ArmHandler.paths.append(path)
        payload = DEEPGRAM_RESPONSE if path == LISTEN_PATH else OPENAI_RESPONSE
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture()
def arm_server():
    _ArmHandler.paths = []
    server = HTTPServer(("127.0.0.1", 0), _ArmHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


@dataclass
class Workload:
    """A one-item workload with a reference, so runs write a hypothesis STM."""

    items: list[dict[str, Any]]
    out_dir: Path

    @property
    def hyp(self) -> str:
        """The hypothesis STM the run left on disk."""
        return (self.out_dir / "hyp" / "item1.hyp.stm").read_text()

    @property
    def manifest(self) -> Any:
        return json.loads((self.out_dir / "manifest.json").read_text())


@pytest.fixture()
def workload(tmp_path: Path) -> Workload:
    audio = tmp_path / "item1.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 200)
    ref = tmp_path / "item1.ref.stm"
    ref.write_text("item1 1 1 0.000 2.000 hola mundo si claro\n")
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    return Workload(
        items=[{"item_id": "item1", "audio_path": audio, "ref_stm_path": ref}],
        out_dir=out_dir,
    )


class TestQualityWorkloadTransportSelection:
    def test_default_run_still_uses_the_openai_endpoint(self, arm_server, workload):
        """Byte-for-byte the existing behaviour: no flag, no change."""
        from coro.bench.orchestrate import run_workload

        run_workload(
            items=workload.items,
            base_url=arm_server,
            out_dir=workload.out_dir,
            reps=1,
            subcommand="quality-artifacts-only",
        )

        assert _ArmHandler.paths == [OPENAI_PATH]
        assert workload.hyp == "item1 1 1 0.000 2.000 hola mundo si claro\n"

    def test_deepgram_run_scores_per_word_speakers(self, arm_server, workload):
        """The point of the whole change: three runs, not one summarised line."""
        from coro.bench.orchestrate import run_workload

        run_workload(
            items=workload.items,
            base_url=arm_server,
            out_dir=workload.out_dir,
            reps=1,
            subcommand="quality-artifacts-only",
            deepgram=True,
        )

        assert _ArmHandler.paths == [LISTEN_PATH]
        assert workload.hyp.splitlines() == [
            "item1 1 1 0.000 1.000 hola mundo",
            "item1 1 2 1.200 1.600 si",
            "item1 1 -1 1.600 2.000 claro",
        ]

    def test_the_two_arms_disagree_about_abstention(self, arm_server, workload):
        """Why this had to be wired: the default arm cannot see abstention at all."""
        from coro.bench.orchestrate import run_workload

        run_workload(
            items=workload.items,
            base_url=arm_server,
            out_dir=workload.out_dir,
            reps=1,
            subcommand="quality-artifacts-only",
        )
        openai_hyp = workload.hyp

        run_workload(
            items=workload.items,
            base_url=arm_server,
            out_dir=workload.out_dir,
            reps=1,
            subcommand="quality-artifacts-only",
            deepgram=True,
        )
        deepgram_hyp = workload.hyp

        assert "-1" not in openai_hyp
        assert "-1" in deepgram_hyp


class TestManifestRecordsTheScoredSurface:
    """Which endpoint produced a number is provenance, not a detail.

    Two runs of the same workload can differ only by transport and produce
    different WDER for that reason alone, so the artefact has to say which one
    it was.
    """

    def test_manifest_records_a_deepgram_run(self, arm_server, workload):
        from coro.bench.orchestrate import run_workload

        run_workload(
            items=workload.items,
            base_url=arm_server,
            out_dir=workload.out_dir,
            reps=1,
            subcommand="quality-artifacts-only",
            deepgram=True,
        )

        manifest = workload.manifest
        assert manifest["deepgram"] is True

    def test_manifest_records_a_default_run(self, arm_server, workload):
        from coro.bench.orchestrate import run_workload

        run_workload(
            items=workload.items,
            base_url=arm_server,
            out_dir=workload.out_dir,
            reps=1,
            subcommand="quality-artifacts-only",
        )

        manifest = workload.manifest
        assert manifest["deepgram"] is False
