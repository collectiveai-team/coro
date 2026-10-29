"""End-to-end test of the ``all`` subcommand: performance and quality in one run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from coro.bench.models.quality import QualitySummary, ScoreMetrics, ScoreResult
from support.bench_http import RecordingHandler, placeholder_audio, results_dir
from support.network import LOOPBACK

pytestmark = LOOPBACK


def _ref_stm(tmp_path: Path, stem: str, line: str) -> Path:
    path = tmp_path / f"{stem}.ref.stm"
    path.write_text(line)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@contextmanager
def _scoring_stubbed(*item_ids: str) -> Iterator[None]:
    """Replace meeteval scoring with an empty result for every item."""
    with (
        patch(
            "coro.bench.quality.score_item",
            return_value=ScoreResult(metrics=ScoreMetrics(), raw={}),
        ),
        patch(
            "coro.bench.quality.combine_items",
            return_value=QualitySummary(
                workload_set=list(item_ids), n_succeeded=len(item_ids), n_failed=0
            ),
        ),
    ):
        yield


_MEETING1_REF = "meeting1 1 SPEAKER_00 0.000 1.500 hello world\n"


def _item(tmp_path: Path, item_id: str, ref_line: str | None = None) -> dict[str, object]:
    """A workload item with a placeholder upload and, when given, a reference STM."""
    return {
        "item_id": item_id,
        "audio_path": placeholder_audio(tmp_path, item_id),
        "ref_stm_path": _ref_stm(tmp_path, item_id, ref_line) if ref_line else None,
    }


def _run_all(base_url: str, tmp_path: Path, items: list[dict[str, object]], **kwargs) -> Path:
    from coro.bench.orchestrate import run_all_workload

    out_dir = results_dir(tmp_path)
    run_all_workload(items=items, base_url=base_url, out_dir=out_dir, server_pid=1, **kwargs)
    return out_dir


def _run_mixed(base_url: str, tmp_path: Path) -> Path:
    """Two referenced items and one ad-hoc item, two reps each."""
    items = [
        _item(tmp_path, "IB4001", "IB4001 1 SPEAKER_00 0.000 1.500 hello world\n"),
        _item(tmp_path, "IN1001", "IN1001 1 SPEAKER_00 0.000 1.500 test data\n"),
        _item(tmp_path, "adhoc"),
    ]
    with _scoring_stubbed("IB4001", "IN1001"):
        return _run_all(base_url, tmp_path, items, reps=2)


class TestE2EAllSubcommand:
    def test_all_produces_perf_and_quality_artifacts(self, e2e_server, tmp_path: Path):
        items = [_item(tmp_path, "meeting1", _MEETING1_REF)]
        with _scoring_stubbed("meeting1"):
            out_dir = _run_all(e2e_server, tmp_path, items, reps=1)

        assert (out_dir / "responses" / "meeting1_rep1.json").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep1.csv").exists()
        assert (out_dir / "performance" / "summary.json").exists()
        assert (out_dir / "hyp" / "meeting1.hyp.stm").exists()
        assert (out_dir / "ref" / "meeting1.ref.stm").exists()
        assert (out_dir / "manifest.json").exists()

    def test_warmup_sends_request_before_items(self, e2e_server, tmp_path: Path):
        """Asserted at the server, so it holds whichever transport is selected."""
        item = _item(tmp_path, "meeting1")
        warmup_path = placeholder_audio(tmp_path, "warmup", size=100)

        out_dir = _run_all(e2e_server, tmp_path, [item], reps=1, warmup_audio=warmup_path)

        assert RecordingHandler.uploads == [warmup_path.name, "meeting1.wav"]
        assert not any("warmup" in p.name for p in (out_dir / "responses").iterdir())

    def test_quality_scored_from_rep1_only(self, e2e_server, tmp_path: Path):
        items = [_item(tmp_path, "meeting1", _MEETING1_REF)]
        with _scoring_stubbed("meeting1"):
            out_dir = _run_all(e2e_server, tmp_path, items, reps=3)

        assert (out_dir / "responses" / "meeting1_rep1.json").exists()
        assert (out_dir / "responses" / "meeting1_rep2.json").exists()
        assert (out_dir / "responses" / "meeting1_rep3.json").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep1.csv").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep2.csv").exists()
        assert (out_dir / "performance" / "resource_meeting1_rep3.csv").exists()
        assert (out_dir / "hyp" / "meeting1.hyp.stm").exists()
        assert not (out_dir / "hyp" / "meeting1_rep2.hyp.stm").exists()

    def test_adhoc_audio_without_ref_skips_quality(self, e2e_server, tmp_path: Path):
        out_dir = _run_all(e2e_server, tmp_path, [_item(tmp_path, "my")], reps=1)

        assert (out_dir / "responses" / "my_rep1.json").exists()
        assert (out_dir / "performance" / "resource_my_rep1.csv").exists()
        assert (out_dir / "performance" / "summary.json").exists()
        assert not (out_dir / "hyp" / "my.hyp.stm").exists()
        assert not (out_dir / "quality" / "my.json").exists()

        quality_summary = _read_json(out_dir / "quality" / "summary.json")
        assert quality_summary["n_skipped"] == 1
        assert quality_summary["n_succeeded"] == 0


class TestE2EAllMixedItems:
    """Referenced and ad-hoc items in one ``all`` run."""

    def test_every_item_gets_responses_and_resource_csvs(self, e2e_server, tmp_path: Path):
        out_dir = _run_mixed(e2e_server, tmp_path)

        assert (out_dir / "responses" / "IB4001_rep1.json").exists()
        assert (out_dir / "responses" / "IB4001_rep2.json").exists()
        assert (out_dir / "responses" / "IN1001_rep1.json").exists()
        assert (out_dir / "responses" / "adhoc_rep1.json").exists()
        assert (out_dir / "performance" / "resource_IB4001_rep1.csv").exists()
        assert (out_dir / "performance" / "resource_IB4001_rep2.csv").exists()
        assert (out_dir / "performance" / "resource_adhoc_rep1.csv").exists()
        assert (out_dir / "performance" / "summary.json").exists()

    def test_only_referenced_items_get_hypothesis_stms(self, e2e_server, tmp_path: Path):
        out_dir = _run_mixed(e2e_server, tmp_path)

        assert (out_dir / "hyp" / "IB4001.hyp.stm").exists()
        assert (out_dir / "hyp" / "IN1001.hyp.stm").exists()
        assert not (out_dir / "hyp" / "adhoc.hyp.stm").exists()

    def test_only_referenced_items_are_scored(self, e2e_server, tmp_path: Path):
        out_dir = _run_mixed(e2e_server, tmp_path)

        assert (out_dir / "quality" / "IB4001.json").exists()
        assert (out_dir / "quality" / "IN1001.json").exists()
        assert not (out_dir / "quality" / "adhoc.json").exists()
        quality_summary = _read_json(out_dir / "quality" / "summary.json")
        assert quality_summary["n_skipped"] == 1
        assert quality_summary["n_succeeded"] == 2
