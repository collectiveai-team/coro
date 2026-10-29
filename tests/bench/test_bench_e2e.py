"""End-to-end test: stub server + --audio produces expected artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from support.bench_http import placeholder_audio, results_dir
from support.network import LOOPBACK

pytestmark = LOOPBACK


def _ref_stm(tmp_path: Path, stem: str, line: str) -> Path:
    path = tmp_path / f"{stem}.ref.stm"
    path.write_text(line)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


class TestE2EAdhocAudio:
    def test_single_audio_produces_artifacts(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        ref_stm = _ref_stm(tmp_path, "meeting1", "meeting1 1 SPEAKER_00 0.000 1.500 hello world\n")
        out_dir = results_dir(tmp_path)

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

        resp_data = _read_json(resp_dir / "meeting1_rep1.json")
        assert resp_data["task"] == "transcribe"

        hyp_text = (hyp_dir / "meeting1.hyp.stm").read_text()
        assert "meeting1" in hyp_text
        assert "SPEAKER_00" in hyp_text

        ref_text = (ref_dir / "meeting1.ref.stm").read_text()
        assert "hello world" in ref_text

    def test_manifest_contains_expected_fields(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = placeholder_audio(tmp_path, "meeting1")
        ref_stm = _ref_stm(tmp_path, "meeting1", "meeting1 1 SPK_00 0.0 1.0 test\n")
        out_dir = results_dir(tmp_path)

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

        manifest = _read_json(out_dir / "manifest.json")
        assert "timestamp" in manifest
        assert "hostname" in manifest
        assert "cli_args" in manifest
        assert "workload_set" in manifest
        assert "server_health" in manifest
        assert manifest["workload_set"][0]["item_id"] == "meeting1"
        assert manifest["server_health"]["ready"] is True

    def test_multiple_reps_only_one_hyp_ref(self, e2e_server, tmp_path: Path):
        from coro.bench.orchestrate import run_workload

        audio = placeholder_audio(tmp_path, "meeting1")

        ref_stm = _ref_stm(tmp_path, "meeting1", "meeting1 1 SPK_00 0.0 1.0 test\n")

        out_dir = results_dir(tmp_path)

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

        audio = placeholder_audio(tmp_path, "meeting1")

        out_dir = results_dir(tmp_path)

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
