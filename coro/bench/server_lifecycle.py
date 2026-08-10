"""Server lifecycle management for bench-managed and bench-attached modes."""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_health_json(base_url: str) -> Any:
    import urllib.request

    url = f"{base_url}/health"
    with urllib.request.urlopen(url, timeout=5) as resp:
        import json

        return json.loads(resp.read())


def poll_health(
    base_url: str,
    *,
    timeout: float = 600.0,
    interval: float = 1.0,
    proc: subprocess.Popen | None = None,
) -> Any:
    """Block until ``/health`` reports Capability Readiness and Warmup Readiness.

    When ``proc`` is given, a server that exits before becoming ready fails
    immediately rather than burning the whole timeout on a dead process.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            data = _get_health_json(base_url)
            if data.get("ready") and data.get("warmup_ready"):
                return data
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Bench-managed server at {base_url} exited with code {proc.returncode} "
                "before becoming warmup-ready."
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Server at {base_url} did not become warmup-ready within {timeout}s"
            )
        time.sleep(interval)


class BenchAttachedServer:
    def __init__(self, base_url: str, *, pid: int | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.server_pid = pid

    def __enter__(self) -> BenchAttachedServer:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class BenchManagedServer:
    def __init__(
        self,
        *,
        asr_backend: str = "faster-whisper",
        asr_model: str = "openai/whisper-medium",
        diar_backend: str = "nemo",
        diar_model: str | None = None,
        pipeline: str = "full-memory",
        port: int = 0,
        log_path: Path | None = None,
    ) -> None:
        self._asr_backend = asr_backend
        self._asr_model = asr_model
        self._diar_backend = diar_backend
        self._diar_model = diar_model
        self._pipeline = pipeline
        self._requested_port = port
        self._log_path = log_path
        self._port: int | None = None
        self.base_url: str = ""
        self.server_pid: int | None = None
        self._proc: subprocess.Popen | None = None
        self._log_file: IO[bytes] | None = None

    def _build_env(self) -> Mapping[str, str]:
        import os

        env = dict(os.environ)
        env["CORO_BACKEND_ASR"] = self._asr_backend
        env["CORO_MODEL_ASR"] = self._asr_model
        env["CORO_BACKEND_DIARIZATION"] = self._diar_backend
        if self._diar_model is not None:
            env["CORO_MODEL_DIARIZATION"] = self._diar_model
        env["CORO_PIPELINE"] = self._pipeline
        env["CORO_PORT"] = str(self._port or self._requested_port)
        env["CORO_WARMUP"] = "enabled"
        return env

    def _open_output(self) -> int | IO[bytes]:
        """Where the server's stdout/stderr go.

        Never ``subprocess.PIPE``: nothing drains it, and a chatty model backend
        fills the 64 KiB pipe buffer and deadlocks the server mid-run.
        """
        if self._log_path is None:
            return subprocess.DEVNULL
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("wb")
        return self._log_file

    def __enter__(self) -> BenchManagedServer:
        self._port = self._requested_port if self._requested_port != 0 else find_free_port()
        self.base_url = f"http://127.0.0.1:{self._port}"
        env = self._build_env()
        output = self._open_output()
        logger.info("Starting bench-managed server on %s", self.base_url)
        try:
            self._proc = subprocess.Popen(
                ["coro", "--port", str(self._port)],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            self.server_pid = self._proc.pid
            poll_health(self.base_url, proc=self._proc)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


# Either server mode exposes the same two things the bench client needs: where to
# send requests, and which Server Process Tree root to sample.
ServerHandle = BenchAttachedServer | BenchManagedServer
