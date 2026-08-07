"""Tests for the AMI clip Workload Set materializer.

The measurement Workload Set is the 30 meetings of the AMI ES group, each cut
to a 10-minute clip with a rebased Reference STM. ES is load-bearing: every ES
meeting has exactly four participants, matching the default Diarization Model
Selection's hard four-speaker cap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_STM = "meeting 1 A 0.000 5.000 hola mundo\n"


def _fake_cut(src: Path, dst: Path, start: float, duration: float) -> None:
    """Stand in for the ffmpeg cut so no binary and no real audio are needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"RIFF")


def _run(argv: list[str]) -> tuple[MagicMock, MagicMock]:
    """Run the CLI with download and ffmpeg stubbed; return (ensure, cut) mocks."""
    from coro.bench.utils.make_ami_clip_set import main

    with (
        patch("coro.bench.utils.make_ami_clip_set.ensure_audio_and_annotations") as ensure,
        patch("coro.bench.utils.make_ami_clip.cut_audio_clip", side_effect=_fake_cut) as cut,
        patch("coro.bench.ami.ami_meeting_to_stm", return_value=_STM),
        patch.object(sys, "argv", ["make_ami_clip_set", *argv]),
    ):
        main()
    return ensure, cut


def test_default_run_materializes_thirty_es_ten_minute_clips(tmp_path: Path):
    out_dir = tmp_path / "clips"

    _run(["--ami-root", str(tmp_path / "amicorpus"), "--out-dir", str(out_dir)])

    assert len(list(out_dir.glob("*.wav"))) == 30
    assert len(list(out_dir.glob("*.ref.stm"))) == 30
    assert (out_dir / "ES2002a_0_600.wav").exists()
    assert (out_dir / "ES2016b_0_600.ref.stm").exists()


def test_rerun_skips_materialized_meetings_and_does_not_redownload(tmp_path: Path):
    """The baseline and the post-dedup re-measurement must see identical audio."""
    argv = ["--ami-root", str(tmp_path / "amicorpus"), "--out-dir", str(tmp_path / "clips")]
    _run(argv)

    ensure, cut = _run(argv)

    ensure.assert_not_called()
    cut.assert_not_called()


def test_rerun_refreshes_references_without_recutting_audio(tmp_path: Path):
    """A reference-builder fix must reach clips that already exist.

    Clip audio is an immutable fixture — re-cutting it would break the
    identical-audio guarantee between runs. The Reference STM is a derived
    artifact of code that does change, so skipping it whenever the audio is
    present means a corrected builder never applies to the materialized
    workload and the fix is silently invisible.
    """
    out_dir = tmp_path / "clips"
    argv = ["--ami-root", str(tmp_path / "amicorpus"), "--out-dir", str(out_dir)]
    _run(argv)
    stale = out_dir / "ES2002a_0_600.ref.stm"
    stale.write_text("ES2002a_0_600 1 A 0.000 1.000 stale reference\n")

    ensure, cut = _run(argv)

    cut.assert_not_called()
    ensure.assert_not_called()
    assert "stale reference" not in stale.read_text()
    assert "hola mundo" in stale.read_text()


def test_partial_set_only_materializes_the_missing_meetings(tmp_path: Path):
    out_dir = tmp_path / "clips"
    argv = ["--ami-root", str(tmp_path / "amicorpus"), "--out-dir", str(out_dir)]
    _run(argv)
    (out_dir / "ES2002a_0_600.wav").unlink()

    ensure, cut = _run(argv)

    assert cut.call_count == 1
    ensure.assert_called_once()
    assert ensure.call_args.args[0] == ["ES2002a"]


def test_output_is_consumable_as_a_curated_clip_workload(tmp_path: Path):
    from coro.bench.clips import resolve_clip_items

    out_dir = tmp_path / "clips"
    _run(["--ami-root", str(tmp_path / "amicorpus"), "--out-dir", str(out_dir)])

    items = resolve_clip_items(out_dir)

    assert len(items) == 30
    assert items[0]["item_id"] == "ES2002a_0_600"
    assert all(item["ref_stm_path"] is not None for item in items)
    # MeetEval matches reference and hypothesis on the STM session id, and the
    # Hypothesis STM is keyed by item_id — so the reference must be rebased onto
    # the clip stem, not left as the full meeting id.
    assert items[0]["ref_stm_path"].read_text().split()[0] == "ES2002a_0_600"


def test_group_and_clip_length_are_parameters(tmp_path: Path):
    out_dir = tmp_path / "clips"

    _run(
        [
            "--ami-root",
            str(tmp_path / "amicorpus"),
            "--out-dir",
            str(out_dir),
            "--group",
            "IB",
            "--duration",
            "30",
        ]
    )

    assert len(list(out_dir.glob("*.wav"))) == 5
    assert (out_dir / "IB4001_0_30.wav").exists()


def test_clips_default_to_a_flat_directory_under_the_ami_root(tmp_path: Path):
    ami_root = tmp_path / "amicorpus"

    _run(["--ami-root", str(ami_root), "--group", "IB"])

    assert (ami_root / "clips" / "IB4001_0_600.wav").exists()
