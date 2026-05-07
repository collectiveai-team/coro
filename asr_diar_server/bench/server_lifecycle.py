"""Server lifecycle management for bench-managed and bench-attached modes."""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_health_json(base_url: str) -> dict[str, Any]:
    import urllib.request

    url = f"{base_url}/health"
    with urllib.request.urlopen(url, timeout=5) as resp:
        import json

        return json.loads(resp.read())


def poll_health(
    base_url: str,
    *,
    timeout: float = 300.0,
    interval: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            data = _get_health_json(base_url)
            if data.get("ready") and data.get("warmup_ready"):
                return data
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Server at {base_url} did not become "
                f"warmup-ready within {timeout}s"
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
    ) -> None:
        self._asr_backend = asr_backend
        self._asr_model = asr_model
        self._diar_backend = diar_backend
        self._diar_model = diar_model
        self._pipeline = pipeline
        self._requested_port = port
        self._port: int | None = None
        self.base_url: str = ""
        self.server_pid: int | None = None
        self._proc: subprocess.Popen | None = None

    def _build_env(self) -> dict[str, str]:
        import os

        env = dict(os.environ)
        env["ASR_DIAR_BACKEND_ASR"] = self._asr_backend
        env["ASR_DIAR_MODEL_ASR"] = self._asr_model
        env["ASR_DIAR_BACKEND_DIARIZATION"] = self._diar_backend
        if self._diar_model is not None:
            env["ASR_DIAR_MODEL_DIARIZATION"] = self._diar_model
        env["ASR_DIAR_PIPELINE"] = self._pipeline
        env["ASR_DIAR_PORT"] = str(self._port or self._requested_port)
        env["ASR_DIAR_WARMUP"] = "enabled"
        return env

    def __enter__(self) -> BenchManagedServer:
        self._port = self._requested_port if self._requested_port != 0 else find_free_port()
        self.base_url = f"http://127.0.0.1:{self._port}"
        env = self._build_env()
        self._proc = subprocess.Popen(
            ["asr-diar-server", "--port", str(self._port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.server_pid = self._proc.pid
        poll_health(self.base_url)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
