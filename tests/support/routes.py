"""Flatten a FastAPI app's route table.

Since fastapi 0.140.x/starlette 1.x, ``application.include_router(...)``
no longer flattens the included router's routes into ``app.routes`` --
each call instead leaves an opaque ``fastapi.routing._IncludedRouter``
wrapper there (the "avoid flattening ... for OpenAPI" perf refactor).
Route-table assertions that used to walk ``app.routes`` directly (e.g. to
find the one WebSocket route OpenAPI can't describe) need the old flat
view back, so this recurses through each wrapper's ``original_router``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def flat_routes(app: Any) -> Iterator[Any]:
    """Yield every route reachable from `app`, recursing into included routers."""

    def _walk(routes: Any) -> Iterator[Any]:
        for route in routes:
            if type(route).__name__ == "_IncludedRouter":
                yield from _walk(route.original_router.routes)
            else:
                yield route

    yield from _walk(app.routes)
