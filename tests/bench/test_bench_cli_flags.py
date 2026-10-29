"""Tests for the coro-bench flags: server, audio and attached-only combinations."""

from __future__ import annotations

from pathlib import Path

import pytest

from coro.bench.cli import parse_args


class TestCliMutualExclusivity:
    def test_server_url_with_server_asr_backend_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "quality",
                    "--server-url",
                    "http://localhost:8000",
                    "--server-asr-backend",
                    "faster-whisper",
                ]
            )

    def test_server_url_with_server_asr_model_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "quality",
                    "--server-url",
                    "http://localhost:8000",
                    "--server-asr-model",
                    "openai/whisper-medium",
                ]
            )

    def test_server_url_with_server_diar_backend_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "quality",
                    "--server-url",
                    "http://localhost:8000",
                    "--server-diar-backend",
                    "nemo",
                ]
            )

    def test_server_url_with_server_diar_model_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "quality",
                    "--server-url",
                    "http://localhost:8000",
                    "--server-diar-model",
                    "nvidia/diar_streaming_sortformer_4spk-v2",
                ]
            )

    def test_server_url_with_server_pipeline_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "quality",
                    "--server-url",
                    "http://localhost:8000",
                    "--server-pipeline",
                    "full-memory",
                ]
            )

    def test_server_url_with_server_port_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "quality",
                    "--server-url",
                    "http://localhost:8000",
                    "--server-port",
                    "8001",
                ]
            )

    def test_server_url_alone_accepted(self):
        args = parse_args(["quality", "--server-url", "http://localhost:8000"])
        assert args.server_url == "http://localhost:8000"

    def test_server_asr_backend_alone_accepted(self):
        args = parse_args(["quality", "--server-asr-backend", "faster-whisper"])
        assert args.server_asr_backend == "faster-whisper"


class TestCliServerFlags:
    def test_server_asr_backend_default(self):
        args = parse_args(["quality"])
        assert args.server_asr_backend == "faster-whisper"

    def test_server_asr_model_default(self):
        args = parse_args(["quality"])
        assert args.server_asr_model == "openai/whisper-medium"

    def test_server_diar_backend_default(self):
        args = parse_args(["quality"])
        assert args.server_diar_backend == "nemo"

    def test_server_diar_model_default(self):
        args = parse_args(["quality"])
        assert args.server_diar_model == "nvidia/diar_streaming_sortformer_4spk-v2"

    def test_server_pipeline_default(self):
        args = parse_args(["quality"])
        assert args.server_pipeline == "full-memory"

    def test_server_port_default(self):
        args = parse_args(["quality"])
        assert args.server_port == 0

    def test_server_url_default_none(self):
        args = parse_args(["quality"])
        assert args.server_url is None

    def test_no_diarization_sets_diar_backend_none(self):
        args = parse_args(["quality", "--no-diarization"])
        assert args.server_diar_backend == "none"


class TestCliAudioFlags:
    def test_audio_flag_accepted(self, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.touch()
        args = parse_args(["performance", "--audio", str(audio)])
        assert args.audio == audio

    def test_reference_stm_flag_accepted(self, tmp_path: Path):
        stm = tmp_path / "test.stm"
        stm.touch()
        audio = tmp_path / "test.wav"
        audio.touch()
        args = parse_args(["quality", "--audio", str(audio), "--reference-stm", str(stm)])
        assert args.reference_stm == stm

    def test_audio_defaults_none(self):
        args = parse_args(["quality"])
        assert args.audio is None

    def test_reference_stm_defaults_none(self):
        args = parse_args(["quality"])
        assert args.reference_stm is None

    def test_cli_audio_with_reference_stm(self, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.touch()
        ref_stm = tmp_path / "test.stm"
        ref_stm.touch()

        args = parse_args(
            [
                "all",
                "--audio",
                str(audio),
                "--reference-stm",
                str(ref_stm),
                "--server-url",
                "http://127.0.0.1:8000",
            ]
        )
        assert args.audio == audio
        assert args.reference_stm == ref_stm
        assert args.server_url == "http://127.0.0.1:8000"

    def test_cli_audio_without_ref_quality_rejected(self, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.touch()

        with pytest.raises(SystemExit):
            parse_args(["quality", "--audio", str(audio)])

    def test_cli_audio_without_ref_performance_allowed(self, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.touch()

        args = parse_args(["performance", "--audio", str(audio)])
        assert args.audio == audio
        assert args.reference_stm is None

    def test_cli_audio_without_ref_all_allowed(self, tmp_path: Path):
        audio = tmp_path / "test.wav"
        audio.touch()

        args = parse_args(["all", "--audio", str(audio)])
        assert args.audio == audio
        assert args.reference_stm is None


class TestCliAttachedOnlyFlags:
    def test_server_pid_without_server_url_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["performance", "--server-pid", "1234"])

    def test_server_match_without_server_url_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["performance", "--server-match", "coro"])

    def test_server_pid_with_server_url_accepted(self):
        args = parse_args(
            ["performance", "--server-url", "http://localhost:8000", "--server-pid", "1234"]
        )
        assert args.server_pid == 1234

    def test_server_match_defaults_none(self):
        args = parse_args(["performance"])
        assert args.server_match is None

    def test_server_pid_defaults_none(self):
        args = parse_args(["performance"])
        assert args.server_pid is None
