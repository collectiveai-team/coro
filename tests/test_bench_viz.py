"""Tests for the meeteval-viz convenience wrapper (discovery + combination)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from asr_diar_server.bench import viz


def _make_run(tmp_path: Path, sessions: dict[str, tuple[str, str]]) -> Path:
    """Create a quality-run out-dir with ref/ and hyp/ STM files."""
    out = tmp_path / "run"
    (out / "ref").mkdir(parents=True)
    (out / "hyp").mkdir(parents=True)
    for sid, (ref, hyp) in sessions.items():
        (out / "ref" / f"{sid}.ref.stm").write_text(ref)
        (out / "hyp" / f"{sid}.hyp.stm").write_text(hyp)
    return out


class TestDiscoverQualityPairs:
    def test_pairs_sorted_and_complete(self, tmp_path: Path):
        out = _make_run(
            tmp_path,
            {
                "b": ("b 1 A 0.0 1.0 hola\n", "b 1 A 0.0 1.0 hola\n"),
                "a": ("a 1 A 0.0 1.0 hi\n", "a 1 A 0.0 1.0 hi\n"),
            },
        )
        pairs = viz.discover_quality_pairs(out)
        assert [p[0] for p in pairs] == ["a", "b"]

    def test_skips_session_missing_reference(self, tmp_path: Path):
        out = _make_run(tmp_path, {"a": ("a 1 A 0.0 1.0 hi\n", "a 1 A 0.0 1.0 hi\n")})
        (out / "hyp" / "orphan.hyp.stm").write_text("orphan 1 A 0.0 1.0 x\n")
        pairs = viz.discover_quality_pairs(out)
        assert [p[0] for p in pairs] == ["a"]

    def test_missing_hyp_dir_returns_empty(self, tmp_path: Path):
        assert viz.discover_quality_pairs(tmp_path / "nope") == []


class TestCombineSessionStms:
    def test_concatenates_all_sessions_with_newlines(self, tmp_path: Path):
        out = _make_run(
            tmp_path,
            {
                "a": ("a 1 A 0.0 1.0 hi", "a 1 A 0.0 1.0 hi"),  # no trailing newline
                "b": ("b 1 A 0.0 1.0 hola\n", "b 1 A 0.0 1.0 hola\n"),
            },
        )
        pairs = viz.discover_quality_pairs(out)
        combined = viz.combine_session_stms(pairs, out / "viz")
        assert combined is not None
        ref_text = combined[0].read_text()
        assert "a 1 A 0.0 1.0 hi\n" in ref_text
        assert "b 1 A 0.0 1.0 hola\n" in ref_text
        assert len(ref_text.splitlines()) == 2

    def test_no_pairs_returns_none(self, tmp_path: Path):
        assert viz.combine_session_stms([], tmp_path / "viz") is None


class TestMeetevalVizArgv:
    def test_argv_includes_all_alignments_and_paths(self, tmp_path: Path):
        argv = viz.meeteval_viz_argv(
            tmp_path / "r.stm", tmp_path / "h.stm", tmp_path / "viz",
            alignments=["tcp", "cp"],
        )
        assert "html" in argv
        a = argv.index("--alignment")
        assert argv[a + 1 : a + 3] == ["tcp", "cp"]
        assert "-r" in argv and "-h" in argv and "-o" in argv


class TestVisualizeQualityDir:
    def test_combines_and_invokes_meeteval_viz(self, tmp_path: Path):
        out = _make_run(tmp_path, {"a": ("a 1 A 0.0 1.0 hi\n", "a 1 A 0.0 1.0 hi\n")})
        with patch("asr_diar_server.bench.viz.subprocess.run") as run:
            viz_dir = viz.visualize_quality_dir(out, alignments=["tcp", "cp"])
        assert viz_dir == out / "viz"
        assert (out / "viz" / "_combined.ref.stm").exists()
        run.assert_called_once()
        argv = run.call_args.args[0]
        assert "--alignment" in argv and "tcp" in argv and "cp" in argv

    def test_no_pairs_returns_none_without_invoking(self, tmp_path: Path):
        (tmp_path / "ref").mkdir()
        with patch("asr_diar_server.bench.viz.subprocess.run") as run:
            assert viz.visualize_quality_dir(tmp_path) is None
        run.assert_not_called()
