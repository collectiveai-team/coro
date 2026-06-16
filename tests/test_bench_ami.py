"""Tests for AMI selector resolution, auto-download, and STM materialization."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coro.bench.ami import resolve_workload_set


class TestResolveWorkloadSetDefaults:
    def test_no_selectors_defaults_to_sample_preset(self):
        meetings = resolve_workload_set(
            ami_meetings=[], ami_groups=[], ami_preset=None
        )
        assert meetings == ["IB4001", "IN1001"]

    def test_explicit_sample_preset(self):
        meetings = resolve_workload_set(
            ami_meetings=[], ami_groups=[], ami_preset="sample"
        )
        assert meetings == ["IB4001", "IN1001"]

    def test_union_dedupes_meetings_and_preset(self):
        meetings = resolve_workload_set(
            ami_meetings=["IB4001"], ami_groups=[], ami_preset="sample"
        )
        assert meetings == ["IB4001", "IN1001"]

    def test_groups_ib_expands(self):
        from coro.bench.ami import AMI_GROUPS

        meetings = resolve_workload_set(
            ami_meetings=[], ami_groups=["IB"], ami_preset=None
        )
        assert meetings == AMI_GROUPS["IB"]

    def test_preset_full_returns_all(self):
        from coro.bench.ami import AMI_GROUPS

        meetings = resolve_workload_set(
            ami_meetings=[], ami_groups=[], ami_preset="full"
        )
        expected = [m for g in AMI_GROUPS for m in AMI_GROUPS[g]]
        assert meetings == expected

    def test_multiple_groups_union(self):
        meetings = resolve_workload_set(
            ami_meetings=[], ami_groups=["IB", "IN"], ami_preset=None
        )
        assert meetings[:5] == ["IB4001", "IB4002", "IB4003", "IB4004", "IB4005"]
        assert meetings[5] == "IN1001"


class TestCliAmiFlags:
    def test_quality_accepts_ami_flags(self):
        from coro.bench.cli import parse_args

        args = parse_args([
            "quality",
            "--ami-meetings", "IB4001", "IN1001",
            "--ami-groups", "IB",
            "--ami-preset", "sample",
            "--ami-root", "/tmp/ami",
            "--no-download",
        ])
        assert args.ami_meetings == ["IB4001", "IN1001"]
        assert args.ami_groups == ["IB"]
        assert args.ami_preset == "sample"
        assert args.ami_root == Path("/tmp/ami")
        assert args.no_download is True

    def test_performance_accepts_ami_flags(self):
        from coro.bench.cli import parse_args

        args = parse_args(["performance", "--ami-preset", "full"])
        assert args.ami_preset == "full"

    def test_all_accepts_ami_flags(self):
        from coro.bench.cli import parse_args

        args = parse_args(["all", "--ami-meetings", "IB4001"])
        assert args.ami_meetings == ["IB4001"]

    def test_ami_root_defaults_to_amicorpus(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.ami_root == Path("./amicorpus/")

    def test_no_download_defaults_false(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.no_download is False

    def test_ami_meetings_defaults_empty(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.ami_meetings == []

    def test_ami_groups_defaults_empty(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.ami_groups == []

    def test_ami_preset_defaults_none(self):
        from coro.bench.cli import parse_args

        args = parse_args(["quality"])
        assert args.ami_preset is None


class TestEnsureAudioAndAnnotations:
    def test_downloads_missing_audio(self, tmp_path: Path):
        from coro.bench.ami import ensure_audio_and_annotations

        zip_path = tmp_path / "ami_public_manual_1.6.2.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dummy.txt", "test")

        with patch("coro.bench.ami.download_meeting_audio") as mock_dl, \
             patch("coro.bench.ami.download_annotations", return_value=zip_path):
            ensure_audio_and_annotations(["IB4001"], tmp_path)
            mock_dl.assert_called_once_with("IB4001", tmp_path)
            assert (tmp_path / ".ami_annotations_extracted").exists()

    def test_skips_download_when_audio_exists(self, tmp_path: Path):
        from coro.bench.ami import ensure_audio_and_annotations, get_audio_path

        audio = get_audio_path(tmp_path, "IB4001")
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.touch()
        marker = tmp_path / ".ami_annotations_extracted"
        marker.touch()

        with patch("coro.bench.ami.download_meeting_audio") as mock_dl:
            ensure_audio_and_annotations(["IB4001"], tmp_path)
            mock_dl.assert_called_once()

    def test_no_download_raises_on_missing_audio(self, tmp_path: Path):
        from coro.bench.ami import ensure_audio_and_annotations

        marker = tmp_path / ".ami_annotations_extracted"
        marker.touch()

        with pytest.raises(RuntimeError, match="Missing audio.*IB4001"):
            ensure_audio_and_annotations(
                ["IB4001", "IN1001"], tmp_path, no_download=True,
            )

    def test_no_download_raises_on_missing_annotations(self, tmp_path: Path):
        from coro.bench.ami import ensure_audio_and_annotations, get_audio_path

        for m in ["IB4001"]:
            audio = get_audio_path(tmp_path, m)
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.touch()

        with pytest.raises(RuntimeError, match="Missing annotations"):
            ensure_audio_and_annotations(
                ["IB4001"], tmp_path, no_download=True,
            )

    def test_unzips_annotations_once(self, tmp_path: Path):
        from coro.bench.ami import ensure_audio_and_annotations

        zip_path = tmp_path / "ami_public_manual_1.6.2.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("words/dummy.txt", "test")
        zip_path.touch()

        with patch("coro.bench.ami.download_meeting_audio"), \
             patch("coro.bench.ami.download_annotations", return_value=zip_path):
            ensure_audio_and_annotations(["IB4001"], tmp_path)
            assert (tmp_path / ".ami_annotations_extracted").exists()

            with patch("coro.bench.ami.download_annotations") as mock_ann:
                ensure_audio_and_annotations(["IB4001"], tmp_path)
                mock_ann.assert_not_called()


class TestMaterializeReferenceStms:
    def test_writes_stm_file(self, tmp_path: Path):
        from coro.bench.ami import materialize_reference_stms

        with patch("coro.bench.ami.ami_meeting_to_stm", return_value="STM_CONTENT"):
            materialize_reference_stms(["IB4001"], tmp_path)

        stm_path = tmp_path / "stm" / "IB4001.ref.stm"
        assert stm_path.exists()
        assert stm_path.read_text() == "STM_CONTENT"

    def test_skips_existing_stm(self, tmp_path: Path):
        from coro.bench.ami import materialize_reference_stms

        stm_dir = tmp_path / "stm"
        stm_dir.mkdir()
        stm_path = stm_dir / "IB4001.ref.stm"
        stm_path.write_text("EXISTING")

        with patch("coro.bench.ami.ami_meeting_to_stm") as mock_stm:
            materialize_reference_stms(["IB4001"], tmp_path)
            mock_stm.assert_not_called()

        assert stm_path.read_text() == "EXISTING"

    def test_writes_multiple_stm_files(self, tmp_path: Path):
        from coro.bench.ami import materialize_reference_stms

        def fake_stm(root, meeting_id):
            return f"STM_{meeting_id}"

        with patch("coro.bench.ami.ami_meeting_to_stm", side_effect=fake_stm):
            materialize_reference_stms(["IB4001", "IN1001"], tmp_path)

        assert (tmp_path / "stm" / "IB4001.ref.stm").read_text() == "STM_IB4001"
        assert (tmp_path / "stm" / "IN1001.ref.stm").read_text() == "STM_IN1001"


class TestMainIntegration:
    def test_main_quality_resolves_and_stms(self, capsys, tmp_path: Path):
        from coro.bench.cli import main

        zip_path = tmp_path / "ami_public_manual_1.6.2.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dummy.txt", "test")

        with patch.object(sys, "argv", [
            "coro-bench", "quality",
            "--ami-meetings", "IB4001",
            "--ami-root", str(tmp_path),
        ]), patch("coro.bench.ami.download_meeting_audio"), \
           patch("coro.bench.ami.download_annotations",
                 return_value=zip_path), \
           patch("coro.bench.ami.ami_meeting_to_stm", return_value="STM"), \
           patch("coro.bench.cli._run_quality") as mock_quality:
            main()
            mock_quality.assert_called_once()

        assert (tmp_path / "stm" / "IB4001.ref.stm").exists()

    def test_main_no_download_error(self, tmp_path: Path):
        from coro.bench.cli import main

        with patch.object(sys, "argv", [
            "coro-bench", "quality",
            "--ami-meetings", "IB4001",
            "--ami-root", str(tmp_path),
            "--no-download",
        ]):
            with pytest.raises(RuntimeError, match="Missing audio"):
                main()


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="Set RUN_NETWORK_TESTS=1 to enable network tests",
)
class TestNetworkDownload:
    def test_download_ib4001(self, tmp_path: Path):
        from coro.bench.ami import ensure_audio_and_annotations

        ensure_audio_and_annotations(["IB4001"], tmp_path)
        audio = tmp_path / "IB4001" / "audio" / "IB4001.Mix-Headset.wav"
        assert audio.exists()
        assert audio.stat().st_size > 0
