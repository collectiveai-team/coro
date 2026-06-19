"""Tests for server lifecycle management (bench-managed and bench-attached)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
                    "nvidia/diar_sortformer_4spk-v1",
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


class TestBenchAttachedServer:
    def test_returns_given_url(self):
        from coro.bench.server_lifecycle import BenchAttachedServer

        handle = BenchAttachedServer("http://localhost:9999", pid=12345)
        assert handle.base_url == "http://localhost:9999"

    def test_returns_given_pid(self):
        from coro.bench.server_lifecycle import BenchAttachedServer

        handle = BenchAttachedServer("http://localhost:9999", pid=12345)
        assert handle.server_pid == 12345

    def test_context_manager_noop(self):
        from coro.bench.server_lifecycle import BenchAttachedServer

        handle = BenchAttachedServer("http://localhost:9999", pid=12345)
        with handle as h:
            assert h.base_url == "http://localhost:9999"


class TestBenchManagedServer:
    def test_spawns_subprocess_and_polls_health(self):
        from coro.bench.server_lifecycle import BenchManagedServer

        managed = BenchManagedServer(
            asr_backend="faster-whisper",
            asr_model="openai/whisper-medium",
            diar_backend="none",
            diar_model=None,
            pipeline="full-memory",
            port=18888,
        )
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.poll.return_value = None
        popen = "coro.bench.server_lifecycle.subprocess.Popen"
        health = "coro.bench.server_lifecycle.poll_health"
        with patch(popen, return_value=mock_proc), patch(health):
            with managed as handle:
                assert handle.base_url == "http://127.0.0.1:18888"
                assert handle.server_pid == 55555
            mock_proc.terminate.assert_called()
            mock_proc.wait.assert_called()

    def test_terminates_on_exception(self):
        from coro.bench.server_lifecycle import BenchManagedServer

        managed = BenchManagedServer(
            asr_backend="faster-whisper",
            asr_model="openai/whisper-medium",
            diar_backend="none",
            diar_model=None,
            pipeline="full-memory",
            port=18888,
        )
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.poll.return_value = None
        popen = "coro.bench.server_lifecycle.subprocess.Popen"
        health = "coro.bench.server_lifecycle.poll_health"
        with (
            patch(popen, return_value=mock_proc),
            patch(health),
            pytest.raises(RuntimeError),
            managed,
        ):
            raise RuntimeError("boom")
        mock_proc.terminate.assert_called()
        mock_proc.wait.assert_called()

    def test_env_vars_set(self):
        from coro.bench.server_lifecycle import BenchManagedServer

        managed = BenchManagedServer(
            asr_backend="faster-whisper",
            asr_model="openai/whisper-medium",
            diar_backend="nemo",
            diar_model="nvidia/some-model",
            pipeline="streaming",
            port=19999,
        )
        env = managed._build_env()
        assert env["CORO_BACKEND_ASR"] == "faster-whisper"
        assert env["CORO_MODEL_ASR"] == "openai/whisper-medium"
        assert env["CORO_BACKEND_DIARIZATION"] == "nemo"
        assert env["CORO_MODEL_DIARIZATION"] == "nvidia/some-model"
        assert env["CORO_PIPELINE"] == "streaming"
        assert env["CORO_PORT"] == "19999"
        assert env["CORO_WARMUP"] == "enabled"


class TestPollHealth:
    def test_polls_until_ready(self):
        from coro.bench.server_lifecycle import poll_health

        call_count = 0

        def fake_get_json(url):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"ready": False, "warmup_ready": False}
            return {"ready": True, "warmup_ready": True}

        getter = "coro.bench.server_lifecycle._get_health_json"
        with patch(getter, side_effect=fake_get_json):
            poll_health("http://localhost:8000", timeout=5, interval=0.01)
        assert call_count == 3

    def test_raises_on_timeout(self):
        from coro.bench.server_lifecycle import poll_health

        def fake_get_json(url):
            return {"ready": False, "warmup_ready": False}

        getter = "coro.bench.server_lifecycle._get_health_json"
        with patch(getter, side_effect=fake_get_json), pytest.raises(TimeoutError, match="warmup"):
            poll_health("http://localhost:8000", timeout=0.05, interval=0.01)


class TestFindFreePort:
    def test_returns_int(self):
        from coro.bench.server_lifecycle import find_free_port

        port = find_free_port()
        assert isinstance(port, int)
        assert port > 0
