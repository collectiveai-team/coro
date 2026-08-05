"""Tests for Quality Benchmark: MeetEval Metric Set and run-level summary."""

from __future__ import annotations

import json
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coro.bench.models.quality import ScoreError, ScoreResult


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

    def log_message(self, format, *args):
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


class TestPyprojectBenchExtra:
    def test_pyproject_declares_bench_optional_extra(self):
        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = toml_path.read_text()
        assert "[project.optional-dependencies]" in text
        assert "bench = [" in text
        assert "meeteval" in text
        assert "rich" in text


class TestRequireMeeteval:
    def test_require_meeteval_exits_with_helpful_message(self):
        with patch.dict(sys.modules, {"meeteval": None}):
            from coro.bench.quality import _require_meeteval

            with pytest.raises(SystemExit) as exc_info:
                _require_meeteval()
            assert exc_info.value.code == 1

    def test_require_meeteval_returns_module_when_available(self):
        mock_meeteval = MagicMock()
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import _require_meeteval

            result = _require_meeteval()
            assert result is mock_meeteval


def _write_stm_pair(tmp_path: Path, name: str, ref: str, hyp: str) -> tuple[Path, Path]:
    """Write a (ref, hyp) STM pair and return their paths."""
    ref_stm = tmp_path / f"{name}.ref.stm"
    hyp_stm = tmp_path / f"{name}.hyp.stm"
    ref_stm.write_text(ref)
    hyp_stm.write_text(hyp)
    return ref_stm, hyp_stm


class TestTextSchemaMetrics:
    """Every text schema is scored, and the schemas stay independent."""

    def test_score_item_reports_every_text_schema(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 2.0 Um , I'm Okay .\n",
            "m 1 A 0.0 2.0 Um , I'm Okay .\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        for schema in (result.metrics.unpunctuated, result.metrics.whisper_english):
            assert schema is not None
            assert schema.cpwer is not None
            assert schema.orcwer is not None
            assert schema.dicpwer is not None

    def test_schemas_score_the_same_hypothesis_differently(self, tmp_path: Path):
        """Casing and fillers are errors under one schema and invisible to the other."""
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 2.0 Um okay we are done\n",
            "m 1 A 0.0 2.0 Okay we're done\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        normalized = result.metrics.unpunctuated
        leaderboard = result.metrics.whisper_english
        assert normalized is not None and normalized.cpwer is not None
        assert leaderboard is not None and leaderboard.cpwer is not None
        # The leaderboard schema forgives the filler, the case and the contraction.
        assert leaderboard.cpwer.wer == 0.0
        assert normalized.cpwer.wer > 0.0

    def test_diarization_only_reference_skips_every_schema(self, tmp_path: Path):
        from coro.bench.stm import DIARIZATION_ONLY_TEXT

        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            f"m 1 A 0.0 1.0 {DIARIZATION_ONLY_TEXT}\n",
            "m 1 A 0.0 1.0 hello world\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        assert result.metrics.unpunctuated is None
        assert result.metrics.whisper_english is None

    def test_combine_items_and_per_item_carry_leaderboard_metrics(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 2.0 hello world\n",
            "m 1 A 0.0 2.0 hello world\n",
        )

        from coro.bench.quality import combine_items, score_item

        result = score_item(ref_stm, hyp_stm)
        result.session_id = "m"
        summary = combine_items([result])

        assert summary.combined is not None
        assert summary.combined.whisper_english is not None
        assert summary.combined.whisper_english.cpwer is not None
        assert summary.per_item[0].whisper_english_cpwer is not None


class TestScoreItem:
    def test_score_item_returns_all_wer_and_der_metrics(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "meeting1",
            "meeting1 1 A 0.000 1.500 hello world\n",
            "meeting1 1 A 0.000 1.500 hello world\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None
        metrics = result.metrics
        assert metrics.cpwer is not None
        assert metrics.orcwer is not None
        assert metrics.dicpwer is not None
        assert metrics.der is not None
        # Perfect match -> zero WER on every speaker-attributed metric.
        assert metrics.cpwer.wer == 0.0
        assert metrics.orcwer.wer == 0.0
        assert metrics.dicpwer.wer == 0.0

    def test_score_item_wer_metrics_have_full_breakdown(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 1.0 test\n",
            "m 1 A 0.0 1.0 test\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None and result.metrics.cpwer is not None
        cpwer = result.metrics.cpwer
        for key in ("wer", "errors", "length", "insertions", "deletions", "substitutions"):
            assert hasattr(cpwer, key)

    def test_score_item_der_has_full_breakdown(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 1.0 test\n",
            "m 1 A 0.0 1.0 test\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        assert result.metrics is not None and result.metrics.der is not None
        der = result.metrics.der
        for key in ("der", "false_alarm", "missed_detection", "speaker_error", "total_speech"):
            assert hasattr(der, key)

    def test_score_item_returns_error_when_meeteval_raises(self, tmp_path: Path):
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 1.0 test\n",
            "m 1 A 0.0 1.0 test\n",
        )

        mock_meeteval = MagicMock()
        mock_meeteval.wer.cpwer.side_effect = RuntimeError("scoring failed")
        with patch.dict(sys.modules, {"meeteval": mock_meeteval}):
            from coro.bench.quality import score_item

            result = score_item(ref_stm, hyp_stm)

        assert result.metrics is None
        assert result.error is not None
        assert result.error.type == "RuntimeError"
        assert "scoring failed" in result.error.message

    def test_score_item_reports_diarization_sanity(self, tmp_path: Path):
        # Single hyp speaker against a two-speaker reference is degenerate.
        ref_stm, hyp_stm = _write_stm_pair(
            tmp_path,
            "m",
            "m 1 A 0.0 2.0 hello world\nm 1 B 2.0 4.0 foo bar\n",
            "m 1 1 0.0 4.0 hello world foo bar\n",
        )

        from coro.bench.quality import score_item

        result = score_item(ref_stm, hyp_stm)

        diar = result.diarization
        assert diar is not None
        assert diar.ref_speakers == 2
        assert diar.hyp_speakers == 1
        assert diar.degenerate is True


class TestSegmentShapeCounters:
    """Unscored transcript-shape counts derived from the Hypothesis STM."""

    def test_counters_from_hypothesis_stm(self, tmp_path: Path):
        from coro.bench.quality import segment_shape_counters, stm_words_per_segment

        hyp = tmp_path / "rec.hyp.stm"
        hyp.write_text(
            "rec 1 X 0.000 1.000 one\n"
            "rec 1 X 1.000 2.000 two words\n"
            "rec 1 Y 2.000 3.000 three words here\n"
        )

        counters = segment_shape_counters(stm_words_per_segment(hyp))

        assert counters.segment_count == 3
        assert counters.median_words_per_segment == 2.0
        assert counters.single_word_segment_count == 1

    def test_empty_transcript_has_no_segments_and_no_median(self, tmp_path: Path):
        from coro.bench.quality import segment_shape_counters, stm_words_per_segment

        hyp = tmp_path / "empty.hyp.stm"
        hyp.write_text("")

        counters = segment_shape_counters(stm_words_per_segment(hyp))

        assert counters.segment_count == 0
        assert counters.median_words_per_segment is None
        assert counters.single_word_segment_count == 0

    def test_single_segment_transcript(self, tmp_path: Path):
        from coro.bench.quality import segment_shape_counters, stm_words_per_segment

        hyp = tmp_path / "one.hyp.stm"
        hyp.write_text("rec 1 X 0.000 3.000 four words in total\n")

        counters = segment_shape_counters(stm_words_per_segment(hyp))

        assert counters.segment_count == 1
        assert counters.median_words_per_segment == 4.0
        assert counters.single_word_segment_count == 0

    def test_all_single_word_segments_are_fully_counted(self, tmp_path: Path):
        """The shredded transcript cpWER rewards must be visible as a counter."""
        from coro.bench.quality import segment_shape_counters, stm_words_per_segment

        hyp = tmp_path / "shredded.hyp.stm"
        hyp.write_text(
            "rec 1 X 0.000 0.400 one\nrec 1 Y 0.400 0.800 two\nrec 1 X 0.800 1.200 three\n"
        )

        counters = segment_shape_counters(stm_words_per_segment(hyp))

        assert counters.segment_count == 3
        assert counters.median_words_per_segment == 1.0
        assert counters.single_word_segment_count == 3

    def test_score_item_retains_hypothesis_segment_word_counts(self, tmp_path: Path):
        from coro.bench.quality import score_item

        ref, hyp = _write_stm_pair(
            tmp_path,
            "rec",
            "rec 1 X 0.000 2.000 hello world\nrec 1 Y 2.000 4.000 foo\n",
            "rec 1 X 0.000 2.000 hello world\nrec 1 Y 2.000 4.000 foo\n",
        )

        result = score_item(ref, hyp)

        assert result.segment_word_counts == [2, 1]

    def test_combine_items_pools_segments_across_the_workload_set(self, tmp_path: Path):
        """The run-level median pools every segment, not the per-item medians.

        Item A's segments are 2 and 1 words (median 1.5); item B's is 4 words.
        Pooled that is [1, 2, 4] -> median 2.0. A median of per-item medians
        would give 2.75, so this assertion discriminates between the two.
        """
        from coro.bench.quality import combine_items

        a = "A 1 X 0.0 2.0 hello world\nA 1 Y 2.0 4.0 foo\n"
        b = "B 1 X 0.0 2.0 one two three four\n"
        summary = combine_items(
            [_scored(tmp_path, "A", a, a, 4.0), _scored(tmp_path, "B", b, b, 2.0)]
        )

        first = summary.per_item[0].segment_shape
        assert first is not None
        assert first.segment_count == 2
        assert first.median_words_per_segment == 1.5
        assert first.single_word_segment_count == 1

        pooled = summary.segment_shape
        assert pooled is not None
        assert pooled.segment_count == 3
        assert pooled.median_words_per_segment == 2.0
        assert pooled.single_word_segment_count == 1


def _scored(tmp_path: Path, name: str, ref: str, hyp: str, seconds: float) -> ScoreResult:
    """Score a tiny STM pair with real meeteval and tag it like the orchestrator."""
    ref_stm, hyp_stm = _write_stm_pair(tmp_path, name, ref, hyp)
    from coro.bench.quality import score_item

    result = score_item(ref_stm, hyp_stm)
    result.session_id = name
    result.audio_seconds = seconds
    return result


class TestCombineItems:
    def test_combine_items_produces_combined_metrics(self, tmp_path: Path):
        from coro.bench.quality import combine_items

        item_results = [
            _scored(
                tmp_path, "A", "A 1 X 0.0 2.0 hello world\n", "A 1 X 0.0 2.0 hello world\n", 2.0
            ),
            _scored(tmp_path, "B", "B 1 X 0.0 2.0 foo bar\n", "B 1 X 0.0 2.0 foo bar\n", 2.0),
        ]

        summary = combine_items(item_results)

        assert summary.n_succeeded == 2
        assert summary.n_failed == 0
        assert summary.combined is not None
        assert summary.combined.cpwer is not None
        assert summary.combined.der is not None
        assert summary.per_item[0].session_id == "A"
        assert summary.per_item[1].session_id == "B"

    def test_combine_items_aggregates_der_across_all_items(self, tmp_path: Path):
        """Regression: combined DER must reflect every item, not just the first.

        Item A is a perfect single-speaker match (DER 0.0); item B collapses two
        reference speakers onto one hypothesis speaker (DER > 0). The old code
        reported only item A's DER (0.0); the aggregate must be > 0.
        """
        from coro.bench.quality import combine_items

        item_results = [
            _scored(
                tmp_path, "A", "A 1 X 0.0 2.0 hello world\n", "A 1 X 0.0 2.0 hello world\n", 2.0
            ),
            _scored(
                tmp_path,
                "B",
                "B 1 P 0.0 2.0 foo bar\nB 1 Q 2.0 4.0 baz qux\n",
                "B 1 1 0.0 4.0 foo bar baz qux\n",
                4.0,
            ),
        ]

        summary = combine_items(item_results)

        assert item_results[0].metrics is not None and item_results[0].metrics.der is not None
        first_item_der = item_results[0].metrics.der.der
        assert first_item_der == 0.0
        assert summary.combined is not None and summary.combined.der is not None
        assert summary.combined.der.der > 0.0
        assert summary.n_degenerate_diarization == 1

    def test_combine_items_counts_failures(self, tmp_path: Path):
        from coro.bench.quality import combine_items

        good = _scored(tmp_path, "A", "A 1 X 0.0 2.0 hello\n", "A 1 X 0.0 2.0 hello\n", 2.0)
        item_results = [
            good,
            ScoreResult(
                session_id="B",
                metrics=None,
                error=ScoreError(type="RuntimeError", message="fail"),
            ),
        ]

        summary = combine_items(item_results)

        assert summary.n_succeeded == 1
        assert summary.n_failed == 1
        assert len(summary.per_item) == 2
        assert summary.per_item[1].session_id == "B"


class TestDiarizationOnly:
    """Diarization-only references (VoxConverse-style) score DER but omit WER."""

    _REF = "rec 1 spkA 0.000 2.000 <sd>\nrec 1 spkB 2.500 4.000 <sd>\n"
    _HYP = "rec 1 0 0.100 2.100 hola mundo\nrec 1 1 2.600 3.900 que tal\n"

    def test_is_diarization_only_stm_detects_sentinel(self, tmp_path: Path):
        from coro.bench.quality import is_diarization_only_stm

        ref, hyp = _write_stm_pair(tmp_path, "rec", self._REF, self._HYP)
        assert is_diarization_only_stm(ref) is True
        assert is_diarization_only_stm(hyp) is False

    def test_score_item_skips_wer_keeps_der(self, tmp_path: Path):
        from coro.bench.quality import score_item

        ref, hyp = _write_stm_pair(tmp_path, "rec", self._REF, self._HYP)
        result = score_item(ref, hyp)

        assert result.diarization_only is True
        assert result.metrics is not None
        # Only DER is computed for diarization-only references.
        assert result.metrics.der is not None
        assert result.metrics.cpwer is None
        assert result.metrics.orcwer is None
        assert result.metrics.dicpwer is None
        assert result.metrics.unpunctuated is None
        assert result.metrics.der.der >= 0.0

    def test_combine_items_aggregates_der_without_wer(self, tmp_path: Path):
        from coro.bench.quality import combine_items

        result = _scored(tmp_path, "rec", self._REF, self._HYP, 4.0)
        summary = combine_items([result])

        assert summary.n_succeeded == 1
        assert summary.n_failed == 0
        assert summary.combined is not None
        assert summary.combined.cpwer is None
        assert summary.combined.der is not None
        entry = summary.per_item[0]
        assert entry.diarization_only is True
        assert entry.cpwer is None
        assert entry.der is not None

    def test_segment_shape_counters_reported_despite_skipped_wer(self, tmp_path: Path):
        """Counters describe the hypothesis, so they survive a reference with no words."""
        from coro.bench.quality import combine_items

        summary = combine_items([_scored(tmp_path, "rec", self._REF, self._HYP, 4.0)])

        entry = summary.per_item[0]
        assert entry.cpwer is None
        assert entry.segment_shape is not None
        assert entry.segment_shape.segment_count == 2
        assert entry.segment_shape.median_words_per_segment == 2.0
        assert summary.segment_shape is not None
        assert summary.segment_shape.segment_count == 2


class TestCLIFlags:
    def test_der_collar_flag_accepted(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality", "--der-collar", "0.25"])
        assert args.der_collar == 0.25

    def test_der_regions_flag_accepted(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality", "--der-regions", "nooverlap"])
        assert args.der_regions == "nooverlap"

    def test_der_collar_default_is_zero(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.der_collar == 0.0

    def test_der_regions_default_is_all(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.der_regions == "all"

    def test_der_regions_invalid_rejected(self):
        from coro.bench.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["quality", "--der-regions", "invalid"])


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
        assert item_data["audio_seconds"] == 3.5
        assert item_data["metrics"] is not None
        assert "cpwer" in item_data["metrics"]
        assert "der" in item_data["metrics"]
        # Segment Shape Counters ride beside the MeetEval Metric Set, not inside it.
        assert item_data["segment_shape"] == {
            "segment_count": 2,
            "median_words_per_segment": 2.0,
            "single_word_segment_count": 0,
        }

        summary = json.loads((quality_dir / "summary.json").read_text())
        assert summary["n_succeeded"] == 1
        assert summary["n_failed"] == 0
        assert "combined" in summary
        assert "per_item" in summary
        assert summary["per_item"][0]["session_id"] == "meeting1"
        assert summary["per_item"][0]["segment_shape"]["segment_count"] == 2
        assert summary["segment_shape"]["segment_count"] == 2

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

        with patch.object(quality_mod, "score_item", side_effect=fail_on_second):
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
