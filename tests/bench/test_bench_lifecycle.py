"""Tests for server lifecycle management (bench-managed and bench-attached)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coro.bench.cli import build_server_handle, parse_args, require_server_pid
from coro.bench.errors import ServerPidUnresolvedError
from coro.bench.process_lookup import ProcessEntry
from coro.bench.server_lifecycle import (
    BenchAttachedServer,
    BenchManagedServer,
    find_free_port,
    poll_health,
)
from support.bench_http import CANNED_HEALTH, serve
from support.network import LOOPBACK
from support.server_process import (
    FAKE_SERVER_OUTPUT,
    NOT_READY_HEALTH,
    FakeServerLauncher,
    FakeServerProcess,
    scripted_health_handler,
)


def _managed(launcher: FakeServerLauncher, port: int, **kwargs) -> BenchManagedServer:
    return BenchManagedServer(
        asr_backend="faster-whisper",
        asr_model="openai/whisper-medium",
        diar_backend="none",
        diar_model=None,
        pipeline="full-memory",
        port=port,
        launcher=launcher,
        **kwargs,
    )


class TestBuildServerHandle:
    def test_defaults_to_a_bench_managed_server(self):
        assert isinstance(build_server_handle(parse_args(["all"])), BenchManagedServer)

    def test_managed_server_receives_the_cli_selection(self):
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
        assert isinstance(handle, BenchManagedServer)
        env = handle.server_env()
        assert env["CORO_BACKEND_ASR"] == "onnx-asr"
        assert env["CORO_MODEL_DIARIZATION"] == "nvidia/some-model"

    def test_managed_server_enables_diarization_by_default(self):
        """A Quality Benchmark that cannot separate speakers cannot report cpWER."""
        handle = build_server_handle(parse_args(["all"]))
        assert isinstance(handle, BenchManagedServer)
        env = handle.server_env()
        assert env["CORO_BACKEND_DIARIZATION"] == "nemo"
        assert env["CORO_MODEL_DIARIZATION"] == "nvidia/diar_streaming_sortformer_4spk-v2"

    def test_no_diarization_flag_disables_diarization(self):
        handle = build_server_handle(parse_args(["all", "--no-diarization"]))
        assert isinstance(handle, BenchManagedServer)
        assert handle.server_env()["CORO_BACKEND_DIARIZATION"] == "none"

    def test_server_url_builds_a_bench_attached_server(self):
        handle = build_server_handle(
            parse_args(["quality", "--server-url", "http://localhost:8000"])
        )
        assert isinstance(handle, BenchAttachedServer)
        assert handle.base_url == "http://localhost:8000"

    def test_attached_server_uses_explicit_server_pid(self):
        handle = build_server_handle(
            parse_args(
                ["performance", "--server-url", "http://localhost:8000", "--server-pid", "4321"]
            )
        )
        assert handle.server_pid == 4321

    def test_attached_sampling_run_resolves_the_pid_from_server_match(self):
        args = parse_args(
            ["all", "--server-url", "http://localhost:8000", "--server-match", "my-server"]
        )
        processes = [
            ProcessEntry(pid=1, ppid=0, cmdline="/sbin/init"),
            ProcessEntry(pid=9876, ppid=1, cmdline="python -m my-server --port 8000"),
            ProcessEntry(pid=9900, ppid=1, cmdline="bash"),
        ]

        handle = build_server_handle(args, processes=processes)

        assert handle.server_pid == 9876

    def test_attached_quality_run_does_not_need_a_pid(self):
        """The quality subcommand samples nothing, so PID resolution must not run.

        An empty process table makes any resolution raise ServerPidUnresolvedError.
        """
        args = parse_args(["quality", "--server-url", "http://localhost:8000"])

        handle = build_server_handle(args, processes=[])

        assert handle.server_pid is None


class TestRequireServerPid:
    def test_raises_instead_of_sampling_an_unrelated_process(self):
        handle = BenchAttachedServer("http://localhost:8000", pid=None)
        with pytest.raises(ServerPidUnresolvedError):
            require_server_pid(handle)

    def test_never_falls_back_to_the_init_process(self):
        handle = BenchAttachedServer("http://localhost:8000", pid=None)
        try:
            resolved = require_server_pid(handle)
        except ServerPidUnresolvedError:
            resolved = None
        assert resolved != 1

    def test_returns_the_handles_pid(self):
        assert require_server_pid(BenchAttachedServer("http://x", pid=77)) == 77


class TestBenchAttachedServer:
    def test_returns_given_url(self):
        handle = BenchAttachedServer("http://localhost:9999", pid=12345)
        assert handle.base_url == "http://localhost:9999"

    def test_returns_given_pid(self):
        handle = BenchAttachedServer("http://localhost:9999", pid=12345)
        assert handle.server_pid == 12345

    def test_context_manager_noop(self):
        handle = BenchAttachedServer("http://localhost:9999", pid=12345)
        with handle as h:
            assert h.base_url == "http://localhost:9999"


class TestBenchManagedServer:
    pytestmark = LOOPBACK

    def test_starts_the_server_and_stops_it_on_exit(self):
        launcher = FakeServerLauncher(pid=55555)
        port = find_free_port()

        with _managed(launcher, port) as handle:
            assert handle.base_url == f"http://127.0.0.1:{port}"
            assert handle.server_pid == 55555
            assert launcher.launches[0].process.terminated is False

        assert launcher.launches[0].process.terminated is True

    def test_spawns_the_serve_subcommand(self):
        """coro's CLI requires a subcommand (`coro serve --port ...`); a bare
        `coro --port ...` fails immediately with "unknown command '--port'".
        """
        launcher = FakeServerLauncher()
        port = find_free_port()

        with _managed(launcher, port):
            pass

        assert launcher.launches[0].argv == ["coro", "serve", "--port", str(port)]

    def test_terminates_on_exception(self):
        launcher = FakeServerLauncher()

        with pytest.raises(RuntimeError, match="boom"), _managed(launcher, find_free_port()):
            raise RuntimeError("boom")

        assert launcher.launches[0].process.terminated is True

    def test_server_starts_with_the_selection_in_its_env(self):
        launcher = FakeServerLauncher()
        port = find_free_port()
        managed = BenchManagedServer(
            asr_backend="faster-whisper",
            asr_model="openai/whisper-medium",
            diar_backend="nemo",
            diar_model="nvidia/some-model",
            pipeline="streaming",
            port=port,
            launcher=launcher,
        )

        with managed:
            pass

        env = launcher.launches[0].env
        assert env["CORO_BACKEND_ASR"] == "faster-whisper"
        assert env["CORO_MODEL_ASR"] == "openai/whisper-medium"
        assert env["CORO_BACKEND_DIARIZATION"] == "nemo"
        assert env["CORO_MODEL_DIARIZATION"] == "nvidia/some-model"
        assert env["CORO_PIPELINE"] == "streaming"
        assert env["CORO_PORT"] == str(port)
        assert env["CORO_WARMUP"] == "enabled"


class TestBenchManagedServerOutput:
    pytestmark = LOOPBACK

    def test_never_uses_an_undrained_pipe(self, tmp_path: Path):
        """An undrained PIPE deadlocks a chatty backend once the buffer fills."""
        launcher = FakeServerLauncher()

        with _managed(launcher, find_free_port(), log_path=tmp_path / "server.log"):
            pass

        launch = launcher.launches[0]
        assert launch.stdout is not subprocess.PIPE
        assert launch.stderr is subprocess.STDOUT

    def test_writes_server_output_to_the_log_path(self, tmp_path: Path):
        log_path = tmp_path / "nested" / "server.log"

        with _managed(FakeServerLauncher(), find_free_port(), log_path=log_path):
            pass

        assert log_path.read_bytes() == FAKE_SERVER_OUTPUT

    def test_tears_down_when_the_server_never_becomes_ready(self, tmp_path: Path):
        launcher = FakeServerLauncher(exit_code=1)
        managed = _managed(launcher, find_free_port(), log_path=tmp_path / "server.log")

        with pytest.raises(RuntimeError, match="exited with code 1"):
            managed.__enter__()

        assert launcher.launches[0].process.terminated is True


class TestPollHealth:
    pytestmark = LOOPBACK

    def test_fails_fast_when_the_server_process_exits(self):
        dead = FakeServerProcess(55555, exit_code=1, server=None)
        url = f"http://127.0.0.1:{find_free_port()}"

        with pytest.raises(RuntimeError, match="exited with code 1"):
            poll_health(url, timeout=30, interval=0.01, proc=dead)

    def test_polls_until_ready(self):
        handler = scripted_health_handler([NOT_READY_HEALTH, NOT_READY_HEALTH, CANNED_HEALTH])

        with serve(handler) as url:
            health = poll_health(url, timeout=5, interval=0.01)

        assert health == CANNED_HEALTH
        assert handler.served == 3

    def test_raises_on_timeout(self):
        handler = scripted_health_handler([NOT_READY_HEALTH])

        with serve(handler) as url, pytest.raises(TimeoutError, match="warmup"):
            poll_health(url, timeout=0.05, interval=0.01)


class TestFindFreePort:
    pytestmark = LOOPBACK

    def test_returns_int(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert port > 0
