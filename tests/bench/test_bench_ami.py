"""Tests for AMI selector resolution, auto-download, and STM materialization."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from coro.bench.ami import (
    ensure_audio_and_annotations,
    get_audio_path,
    materialize_reference_stms,
    resolve_workload_set,
)
from coro.bench.cli import main, parse_args, resolve_meetings
from support.ami_downloads import (
    InMemoryAmiDownloads,
    reference_stm,
    write_annotations,
)


class TestResolveWorkloadSetDefaults:
    def test_no_selectors_defaults_to_sample_preset(self):
        meetings = resolve_workload_set(ami_meetings=[], ami_groups=[], ami_preset=None)
        assert meetings == ["IB4001", "IN1001"]

    def test_explicit_sample_preset(self):
        meetings = resolve_workload_set(ami_meetings=[], ami_groups=[], ami_preset="sample")
        assert meetings == ["IB4001", "IN1001"]

    def test_union_dedupes_meetings_and_preset(self):
        meetings = resolve_workload_set(ami_meetings=["IB4001"], ami_groups=[], ami_preset="sample")
        assert meetings == ["IB4001", "IN1001"]

    def test_groups_ib_expands(self):
        from coro.bench.ami import AMI_GROUPS

        meetings = resolve_workload_set(ami_meetings=[], ami_groups=["IB"], ami_preset=None)
        assert meetings == AMI_GROUPS["IB"]

    def test_preset_full_returns_all(self):
        from coro.bench.ami import AMI_GROUPS

        meetings = resolve_workload_set(ami_meetings=[], ami_groups=[], ami_preset="full")
        expected = [m for g in AMI_GROUPS for m in AMI_GROUPS[g]]
        assert meetings == expected

    def test_multiple_groups_union(self):
        meetings = resolve_workload_set(ami_meetings=[], ami_groups=["IB", "IN"], ami_preset=None)
        assert meetings[:5] == ["IB4001", "IB4002", "IB4003", "IB4004", "IB4005"]
        assert meetings[5] == "IN1001"


class TestCliAmiFlags:
    def test_quality_accepts_ami_flags(self):
        from coro.bench.cli import parse_args

        args = parse_args(
            [
                "quality",
                "--ami-meetings",
                "IB4001",
                "IN1001",
                "--ami-groups",
                "IB",
                "--ami-preset",
                "sample",
                "--ami-root",
                "/tmp/ami",
                "--no-download",
            ]
        )
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


def _local_audio(ami_root: Path, meeting_id: str) -> Path:
    """Write a local audio file of ``meeting_id`` with the bytes ``LOCAL``."""
    audio = get_audio_path(ami_root, meeting_id)
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"LOCAL")
    return audio


def _existing_stm(ami_root: Path, text: str) -> Path:
    """Write a Reference STM of IB4001 holding ``text``, as a past run left it."""
    stm_path = ami_root / "stm" / "IB4001.ref.stm"
    stm_path.parent.mkdir(parents=True, exist_ok=True)
    stm_path.write_text(text)
    return stm_path


class TestEnsureAudioAndAnnotations:
    def test_downloads_missing_audio(self, tmp_path: Path):
        ensure_audio_and_annotations(["IB4001"], tmp_path, downloads=InMemoryAmiDownloads())

        assert get_audio_path(tmp_path, "IB4001").exists()

    def test_existing_audio_is_kept(self, tmp_path: Path):
        audio = _local_audio(tmp_path, "IB4001")

        ensure_audio_and_annotations(["IB4001"], tmp_path, downloads=InMemoryAmiDownloads())

        assert audio.read_bytes() == b"LOCAL"

    def test_no_download_raises_on_missing_audio(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match=r"Missing audio.*IB4001"):
            ensure_audio_and_annotations(
                ["IB4001", "IN1001"],
                tmp_path,
                no_download=True,
            )

    def test_no_download_raises_on_missing_annotations(self, tmp_path: Path):
        _local_audio(tmp_path, "IB4001")

        with pytest.raises(RuntimeError, match="Missing annotations"):
            ensure_audio_and_annotations(
                ["IB4001"],
                tmp_path,
                no_download=True,
            )

    def test_extracts_the_annotations_archive(self, tmp_path: Path):
        ensure_audio_and_annotations(["IB4001"], tmp_path, downloads=InMemoryAmiDownloads())

        assert (tmp_path / "words" / "IB4001.A.words.xml").exists()

    def test_downloads_the_annotations_archive_once(self, tmp_path: Path):
        downloads = InMemoryAmiDownloads()

        ensure_audio_and_annotations(["IB4001"], tmp_path, downloads=downloads)
        ensure_audio_and_annotations(["IB4001"], tmp_path, downloads=downloads)

        assert downloads.annotation_downloads == 1

    def test_no_download_run_uses_the_annotations_of_an_earlier_run(self, tmp_path: Path):
        ensure_audio_and_annotations(["IB4001"], tmp_path, downloads=InMemoryAmiDownloads())

        ensure_audio_and_annotations(["IB4001"], tmp_path, no_download=True)

        assert (tmp_path / "words" / "IB4001.A.words.xml").exists()


class TestMaterializeReferenceStms:
    def test_writes_stm_file(self, tmp_path: Path):
        write_annotations(tmp_path, ["IB4001"])

        materialize_reference_stms(["IB4001"], tmp_path)

        stm_path = tmp_path / "stm" / "IB4001.ref.stm"
        assert stm_path.read_text() == reference_stm("IB4001")

    def test_regenerates_existing_stm_by_default(self, tmp_path: Path):
        """A stale Reference STM is rebuilt, not silently trusted."""
        write_annotations(tmp_path, ["IB4001"])
        stm_path = _existing_stm(tmp_path, "STALE")

        materialize_reference_stms(["IB4001"], tmp_path)

        assert stm_path.read_text() == reference_stm("IB4001")

    def test_reuses_existing_stm_with_explicit_opt_in(self, tmp_path: Path):
        write_annotations(tmp_path, ["IB4001"])
        stm_path = _existing_stm(tmp_path, "EXISTING")

        materialize_reference_stms(["IB4001"], tmp_path, reuse_existing=True)

        assert stm_path.read_text() == "EXISTING"

    def test_opt_in_reuse_still_writes_a_missing_stm(self, tmp_path: Path):
        write_annotations(tmp_path, ["IB4001"])

        materialize_reference_stms(["IB4001"], tmp_path, reuse_existing=True)

        assert (tmp_path / "stm" / "IB4001.ref.stm").read_text() == reference_stm("IB4001")

    def test_writes_multiple_stm_files(self, tmp_path: Path):
        write_annotations(tmp_path, ["IB4001", "IN1001"])

        materialize_reference_stms(["IB4001", "IN1001"], tmp_path)

        assert (tmp_path / "stm" / "IB4001.ref.stm").read_text() == reference_stm("IB4001")
        assert (tmp_path / "stm" / "IN1001.ref.stm").read_text() == reference_stm("IN1001")


def _downloaded_corpus(ami_root: Path, meetings: list[str]) -> Path:
    """Fill ``ami_root`` with the audio and annotations of ``meetings``, as a past run did."""
    ensure_audio_and_annotations(meetings, ami_root, downloads=InMemoryAmiDownloads(meetings))
    return ami_root


def _resolve(ami_root: Path, *flags: str) -> list[str]:
    args = parse_args(
        [
            "quality",
            "--ami-meetings",
            "IB4001",
            "--ami-root",
            str(ami_root),
            "--no-download",
            *flags,
        ]
    )
    return resolve_meetings(args)


class TestResolveMeetings:
    def test_resolves_the_meetings_and_writes_their_reference_stms(self, tmp_path: Path):
        ami_root = _downloaded_corpus(tmp_path, ["IB4001"])

        meetings = _resolve(ami_root)

        assert meetings == ["IB4001"]
        assert (ami_root / "stm" / "IB4001.ref.stm").read_text() == reference_stm("IB4001")

    def test_reuse_opt_in_keeps_the_existing_reference_stm(self, tmp_path: Path):
        ami_root = _downloaded_corpus(tmp_path, ["IB4001"])
        stm_path = _existing_stm(ami_root, "FROZEN")

        _resolve(ami_root, "--reuse-reference-stms")

        assert stm_path.read_text() == "FROZEN"

    def test_regenerates_reference_stms_by_default(self, tmp_path: Path):
        ami_root = _downloaded_corpus(tmp_path, ["IB4001"])
        stm_path = _existing_stm(ami_root, "STALE")

        _resolve(ami_root)

        assert stm_path.read_text() == reference_stm("IB4001")


class TestMainIntegration:
    def test_main_no_download_error(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Missing audio"):
            main(
                [
                    "quality",
                    "--ami-meetings",
                    "IB4001",
                    "--ami-root",
                    str(tmp_path),
                    "--no-download",
                ]
            )


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
