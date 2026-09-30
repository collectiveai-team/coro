"""A real uvicorn server on a real port, for WebSocket end-to-end tests.

Starlette's in-process ``TestClient`` never exercises uvicorn's WebSocket
protocol, so transport behaviour (keepalive pings, read pausing) is only
observable against a bound socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading

import uvicorn


def keep_injected_runtime(app, settings):
    """Stop ``app``'s lifespan from replacing the injected runtime with real models.

    ``create_app``'s lifespan builds an ASR adapter and runs warmup, which would
    load a real model; transport tests keep the fakes set on ``app.state``.
    """
    runtime = app.state.runtime

    @contextlib.asynccontextmanager
    async def _lifespan(application):
        application.state.settings = settings
        application.state.runtime = runtime
        yield

    app.router.lifespan_context = _lifespan
    return app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveServer:
    """Run ``app`` under uvicorn in a thread; extra kwargs go to ``uvicorn.Config``."""

    def __init__(self, app, **config) -> None:
        self.port = free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", **config)
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    async def __aenter__(self):
        self._thread.start()
        for _ in range(200):
            if self._server.started:
                return self
            await asyncio.sleep(0.05)
        raise RuntimeError("uvicorn did not start")

    async def __aexit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/listen"

    async def wait_for_handlers(self, *, attempts: int = 400) -> None:
        """Wait until every ASGI handler task has returned, or raise."""
        for _ in range(attempts):
            if not self._server.server_state.tasks:
                return
            await asyncio.sleep(0.025)
        raise RuntimeError("ASGI handlers still running")
