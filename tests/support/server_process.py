"""An in-memory adapter of the server launcher port, and scripted ``/health`` servers.

The fake launcher stands in for ``coro serve``: it answers ``/health`` on the
port of its command line from a thread, so the bench polls a real socket.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import HTTPServer
from typing import IO, ClassVar

from support.bench_http import CANNED_HEALTH, StubHandler

NOT_READY_HEALTH = {**CANNED_HEALTH, "ready": False, "warmup_ready": False}

FAKE_SERVER_OUTPUT = b"fake coro serve started\n"


def scripted_health_handler(bodies: Sequence[dict[str, object]]) -> type[StubHandler]:
    """A handler answering ``/health`` with ``bodies`` in order, then the last one forever.

    The returned class counts the ``/health`` requests it served in ``served``.
    """

    class ScriptedHealthHandler(StubHandler):
        served: ClassVar[int] = 0

        def do_GET(self):
            cls = type(self)
            self.health = bodies[min(cls.served, len(bodies) - 1)]
            cls.served += 1
            super().do_GET()

    return ScriptedHealthHandler


class FakeServerProcess:
    """A server process that runs until terminated, or one that has already exited."""

    def __init__(self, pid: int, *, exit_code: int | None, server: HTTPServer | None) -> None:
        self.pid = pid
        self.returncode = exit_code
        self.terminated = False
        self._server = server

    def poll(self) -> int | None:
        """Return the exit code once the process has exited, else None."""
        return self.returncode

    def terminate(self) -> None:
        """Stop answering ``/health`` and exit with SIGTERM's code."""
        self.terminated = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        """Stop the process as :meth:`terminate` does."""
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        """Return the exit code; the fake has exited once terminated."""
        return self.returncode if self.returncode is not None else 0


@dataclass(frozen=True)
class Launch:
    """One server start: its command line, environment and output targets."""

    argv: list[str]
    env: Mapping[str, str]
    stdout: int | IO[bytes]
    stderr: int
    process: FakeServerProcess


class FakeServerLauncher:
    """The server launcher port without a child process.

    A started server answers a ready ``/health`` on its ``--port`` and writes
    :data:`FAKE_SERVER_OUTPUT` to its output. With ``exit_code`` set, the server
    exits at once and listens on nothing, as a server that crashes on startup.
    """

    def __init__(self, *, exit_code: int | None = None, pid: int = 55555) -> None:
        self.exit_code = exit_code
        self.pid = pid
        self.launches: list[Launch] = []

    def __call__(
        self, argv: list[str], *, env: Mapping[str, str], stdout: int | IO[bytes], stderr: int
    ) -> FakeServerProcess:
        """Start answering ``/health`` on the port of ``argv`` and return the process."""
        server = None
        if self.exit_code is None:
            port = int(argv[argv.index("--port") + 1])
            server = HTTPServer(("127.0.0.1", port), StubHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        if not isinstance(stdout, int):
            stdout.write(FAKE_SERVER_OUTPUT)
            stdout.flush()
        process = FakeServerProcess(self.pid, exit_code=self.exit_code, server=server)
        self.launches.append(Launch(list(argv), dict(env), stdout, stderr, process))
        return process
