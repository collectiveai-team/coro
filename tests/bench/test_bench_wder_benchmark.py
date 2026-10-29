"""Tests that WDER reaches the Quality Benchmark: item scores, summaries and the report."""

from __future__ import annotations

import pytest

from coro.bench.wder import UNKNOWN_SPEAKER_LABEL
from support.wder_stm import REF

pytest.importorskip("meeteval")


def _markdown_table(markdown: str, heading_fragment: str) -> list[str]:
    """Return the pipe-table lines under the heading containing ``heading_fragment``."""
    lines = markdown.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("#") and heading_fragment in line
    )
    tail = lines[start + 1 :]
    end = next((i for i, line in enumerate(tail) if line.startswith("#")), len(tail))
    return [line for line in tail[:end] if line.startswith("|")]


_ABSTAINING_HYP = [
    ("S1", 0.0, 1.0, "alpha"),
    (UNKNOWN_SPEAKER_LABEL, 1.0, 2.0, "bravo"),
    ("S1", 2.0, 3.0, "charlie"),
    ("S2", 3.0, 4.0, "delta"),
]

_PUNCTUATED_HYP = [
    ("S1", 0.0, 1.0, "Alpha,"),
    ("S2", 1.0, 2.0, "bravo."),
    ("S1", 2.0, 3.0, "Charlie!"),
    ("S2", 3.0, 4.0, "delta"),
]


def _scored_session(stm_pair, hyp_segments):
    """Score one 4 s session ``sess`` against REF, as the Quality Benchmark does."""
    from coro.bench.quality import score_item

    ref, hyp = stm_pair(REF, hyp_segments)
    result = score_item(ref, hyp)
    result.session_id = "sess"
    result.audio_seconds = 4.0
    return result


def _render_report(summary, tmp_path) -> str:
    """Write ``summary`` where the report reads it and render the markdown."""
    import dataclasses
    import json

    from coro.bench.report import build_report
    from coro.bench.report_markdown import render_markdown

    out_dir = tmp_path / "out"
    (out_dir / "quality").mkdir(parents=True)
    (out_dir / "quality" / "summary.json").write_text(json.dumps(dataclasses.asdict(summary)))
    return render_markdown(build_report(out_dir))


class TestBenchmarkIntegration:
    def test_three_numbers_reach_the_item_score(self, stm_pair):
        """Per-item scores carry WDER in both lanes."""
        result = _scored_session(stm_pair, _ABSTAINING_HYP)

        assert result.metrics is not None
        assert result.metrics.wder is not None
        assert result.metrics.normalized is not None
        assert result.metrics.normalized.wder is not None

    def test_three_numbers_reach_the_summary(self, stm_pair):
        """The per-item entries and the combined summary carry WDER."""
        from coro.bench.quality import combine_items

        result = _scored_session(stm_pair, _ABSTAINING_HYP)
        summary = combine_items([result])

        entry = summary.per_item[0]
        assert entry.wder is not None
        assert entry.abstention_rate == pytest.approx(0.25, abs=1e-12)
        assert summary.combined is not None
        assert summary.combined.wder is not None
        assert result.metrics is not None
        assert result.metrics.wder is not None
        assert summary.combined.wder.scored == result.metrics.wder.scored

    def test_three_numbers_reach_the_report(self, stm_pair, tmp_path):
        """The rendered markdown carries the WDER section."""
        from coro.bench.quality import combine_items

        summary = combine_items([_scored_session(stm_pair, _ABSTAINING_HYP)])
        markdown = _render_report(summary, tmp_path)

        assert "Speaker Attribution (WDER)" in markdown
        assert "WDER-claimed" in markdown

    def test_existing_metrics_are_not_displaced(self, stm_pair):
        """No metric is removed: cpWER, ORC-WER, DI-cpWER and DER all survive."""
        from coro.bench.quality import score_item

        ref, hyp = stm_pair(REF, [("S1", 0.0, 4.0, "alpha bravo charlie delta")])
        metrics = score_item(ref, hyp).metrics
        assert metrics is not None
        assert metrics.cpwer is not None
        assert metrics.orcwer is not None
        assert metrics.dicpwer is not None
        assert metrics.der is not None

    def test_normalized_lane_scores_its_own_wder(self, stm_pair):
        """The two lanes score different word sets, so the normalized lane needs a
        WDER of its own. Losing it is silent: the raw lane still reports one."""
        result = _scored_session(stm_pair, _PUNCTUATED_HYP)

        assert result.metrics is not None
        assert result.metrics.normalized is not None
        assert result.metrics.normalized.wder is not None

    def test_normalized_lane_wder_reaches_the_summary(self, stm_pair):
        from coro.bench.quality import combine_items

        summary = combine_items([_scored_session(stm_pair, _PUNCTUATED_HYP)])

        assert summary.per_item[0].normalized_wder is not None
        assert summary.combined is not None
        assert summary.combined.normalized is not None
        assert summary.combined.normalized.wder is not None

    def test_normalized_lane_wder_fills_its_table(self, stm_pair, tmp_path):
        """A lost normalized WDER leaves the normalized table's declared columns
        and its populated cells disagreeing."""
        from coro.bench.quality import combine_items

        summary = combine_items([_scored_session(stm_pair, _PUNCTUATED_HYP)])
        table = _markdown_table(_render_report(summary, tmp_path), "Normalized Metric Lane")

        header, separator, *rows = table
        assert "WDER" in header
        # One scored session plus the combined row, every one as wide as the header.
        cells = len(header.split("|"))
        assert len(separator.split("|")) == cells
        assert [len(row.split("|")) for row in rows] == [cells, cells]

    def test_skipped_normalized_lane_leaves_the_raw_lane_wder_intact(self, stm_pair):
        """A normalized lane with nothing left to score must not void raw-lane WDER."""
        from coro.bench.quality import combine_items, score_item

        ref, hyp = stm_pair(REF, [("S1", 0.0, 4.0, "[tos]")])
        result = score_item(ref, hyp)
        result.session_id = "sess"

        assert result.metrics is not None
        assert result.metrics.normalized is None
        assert result.metrics.wder is not None

        summary = combine_items([result])
        assert summary.combined is not None
        assert summary.combined.wder is not None
        assert summary.combined.normalized is not None
        assert summary.combined.normalized.wder is None
