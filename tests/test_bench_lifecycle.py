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


class TestBuildServerHandle:
    def test_defaults_to_a_bench_managed_server(self):
        from coro.bench.cli import build_server_handle
        from coro.bench.server_lifecycle import BenchManagedServer

        assert isinstance(build_server_handle(parse_args(["all"])), BenchManagedServer)

    def test_managed_server_receives_the_cli_selection(self):
        from coro.bench.cli import build_server_handle

        handle = build_server_handle(
            parse_args(
                [
                    "all",
                    "--server-asr-backend",
                    "onnx-asr",
                    "--server-diar-model",
                    "nvidia/some-model",
                ]
            )
        )
        env = handle._build_env()
        assert env["CORO_BACKEND_ASR"] == "onnx-asr"
        assert env["CORO_MODEL_DIARIZATION"] == "nvidia/some-model"

    def test_managed_server_enables_diarization_by_default(self):
        """A Quality Benchmark that cannot separate speakers cannot report cpWER."""
        from coro.bench.cli import build_server_handle

        env = build_server_handle(parse_args(["all"]))._build_env()
        assert env["CORO_BACKEND_DIARIZATION"] == "nemo"
        assert env["CORO_MODEL_DIARIZATION"]

    def test_no_diarization_flag_disables_diarization(self):
        from coro.bench.cli import build_server_handle

        env = build_server_handle(parse_args(["all", "--no-diarization"]))._build_env()
        assert env["CORO_BACKEND_DIARIZATION"] == "none"

    def test_server_url_builds_a_bench_attached_server(self):
        from coro.bench.cli import build_server_handle
        from coro.bench.server_lifecycle import BenchAttachedServer

        handle = build_server_handle(
            parse_args(["quality", "--server-url", "http://localhost:8000"])
        )
        assert isinstance(handle, BenchAttachedServer)
        assert handle.base_url == "http://localhost:8000"

    def test_attached_server_uses_explicit_server_pid(self):
        from coro.bench.cli import build_server_handle

        handle = build_server_handle(
            parse_args(
                ["performance", "--server-url", "http://localhost:8000", "--server-pid", "4321"]
            )
        )
        assert handle.server_pid == 4321

    def test_attached_sampling_run_resolves_the_pid_from_server_match(self):
        from coro.bench.cli import build_server_handle

        args = parse_args(
            ["all", "--server-url", "http://localhost:8000", "--server-match", "my-server"]
        )
        resolver = "coro.bench.process_lookup.resolve_server_pid"
        with patch(resolver, return_value=9876) as mock_resolve:
            handle = build_server_handle(args)

        assert mock_resolve.call_args.args[0] == "my-server"
        assert handle.server_pid == 9876

    def test_attached_quality_run_does_not_need_a_pid(self):
        """The quality subcommand samples nothing, so PID resolution must not run."""
        from coro.bench.cli import build_server_handle

        args = parse_args(["quality", "--server-url", "http://localhost:8000"])
        resolver = "coro.bench.process_lookup.resolve_server_pid"
        with patch(resolver) as mock_resolve:
            handle = build_server_handle(args)

        mock_resolve.assert_not_called()
        assert handle.server_pid is None


class TestRequireServerPid:
    def test_raises_instead_of_sampling_an_unrelated_process(self):
        from coro.bench.cli import _require_server_pid
        from coro.bench.errors import ServerPidUnresolvedError
        from coro.bench.server_lifecycle import BenchAttachedServer

        handle = BenchAttachedServer("http://localhost:8000", pid=None)
        with pytest.raises(ServerPidUnresolvedError):
            _require_server_pid(handle)

    def test_never_falls_back_to_the_init_process(self):
        from coro.bench.cli import _require_server_pid
        from coro.bench.errors import ServerPidUnresolvedError
        from coro.bench.server_lifecycle import BenchAttachedServer

        handle = BenchAttachedServer("http://localhost:8000", pid=None)
        try:
            resolved = _require_server_pid(handle)
        except ServerPidUnresolvedError:
            resolved = None
        assert resolved != 1

    def test_returns_the_handles_pid(self):
        from coro.bench.cli import _require_server_pid
        from coro.bench.server_lifecycle import BenchAttachedServer

        assert _require_server_pid(BenchAttachedServer("http://x", pid=77)) == 77


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


class TestBenchManagedServerOutput:
    def test_never_uses_an_undrained_pipe(self, tmp_path: Path):
        """An undrained PIPE deadlocks a chatty backend once the buffer fills."""
        import subprocess

        from coro.bench.server_lifecycle import BenchManagedServer

        managed = BenchManagedServer(port=18888, log_path=tmp_path / "server.log")
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.poll.return_value = None
        popen = "coro.bench.server_lifecycle.subprocess.Popen"
        health = "coro.bench.server_lifecycle.poll_health"
        with patch(popen, return_value=mock_proc) as mock_popen, patch(health), managed:
            pass

        kwargs = mock_popen.call_args.kwargs
        assert kwargs["stdout"] is not subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT

    def test_writes_server_output_to_the_log_path(self, tmp_path: Path):
        from coro.bench.server_lifecycle import BenchManagedServer

        log_path = tmp_path / "nested" / "server.log"
        managed = BenchManagedServer(port=18888, log_path=log_path)
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.poll.return_value = None
        popen = "coro.bench.server_lifecycle.subprocess.Popen"
        health = "coro.bench.server_lifecycle.poll_health"
        with patch(popen, return_value=mock_proc), patch(health), managed:
            pass

        assert log_path.exists()

    def test_tears_down_when_the_server_never_becomes_ready(self, tmp_path: Path):
        from coro.bench.server_lifecycle import BenchManagedServer

        managed = BenchManagedServer(port=18888, log_path=tmp_path / "server.log")
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        popen = "coro.bench.server_lifecycle.subprocess.Popen"
        health = "coro.bench.server_lifecycle.poll_health"
        with (
            patch(popen, return_value=mock_proc),
            patch(health, side_effect=TimeoutError("never ready")),
            pytest.raises(TimeoutError),
            managed,
        ):
            pass

        mock_proc.terminate.assert_called()


class TestPollHealth:
    def test_fails_fast_when_the_server_process_exits(self):
        from coro.bench.server_lifecycle import poll_health

        dead = MagicMock()
        dead.poll.return_value = 1
        dead.returncode = 1
        getter = "coro.bench.server_lifecycle._get_health_json"
        with (
            patch(getter, side_effect=ConnectionRefusedError),
            pytest.raises(RuntimeError, match="exited with code 1"),
        ):
            poll_health("http://localhost:8000", timeout=30, interval=0.01, proc=dead)

    def test_polls_until_ready(self):
        from coro.bench.server_lifecycle import poll_health

        call_count = 0

        not_ready = {"ready": False, "warmup_ready": False}
        ready = {"ready": True, "warmup_ready": True}

        def fake_get_json(url):
            nonlocal call_count
            call_count += 1
            return not_ready if call_count < 3 else ready

        getter = "coro.bench.server_lifecycle._get_health_json"
        with patch(getter, side_effect=fake_get_json):
            poll_health("http://localhost:8000", timeout=5, interval=0.01)
        assert call_count == 3

    def test_raises_on_timeout(self):
        from coro.bench.server_lifecycle import poll_health

        not_ready = {"ready": False, "warmup_ready": False}

        def fake_get_json(url):
            return not_ready

        getter = "coro.bench.server_lifecycle._get_health_json"
        with patch(getter, side_effect=fake_get_json), pytest.raises(TimeoutError, match="warmup"):
            poll_health("http://localhost:8000", timeout=0.05, interval=0.01)


class TestFindFreePort:
    def test_returns_int(self):
        from coro.bench.server_lifecycle import find_free_port

        port = find_free_port()
        assert isinstance(port, int)
        assert port > 0
