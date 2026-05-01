r"""OpenAI-Exact SSE helpers.

Public SSE streaming must match OpenAI event framing exactly.
No package-specific progress events are emitted.

Event flow::

  data: {"type": "transcript.text.delta", "delta": "<text>"}\n\n
  ... (zero or more delta events) ...
  data: {"type": "transcript.text.done", "text": "<json_string>"}\n\n
  data: [DONE]\n\n
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _sse_generator(event_source: AsyncIterator[dict[str, Any]]):
    r"""Yield SSE-framed lines from an async event source.

    The event source must yield dicts.  After all events the generator
    emits ``data: [DONE]\n\n``.  On error it emits an error event.
    """
    try:
        async for event in event_source:
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        error_event = json.dumps(
            {
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                }
            }
        )
        yield f"data: {error_event}\n\n"
        yield "data: [DONE]\n\n"


def streaming_response(event_source: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    """Build a StreamingResponse that emits OpenAI-Exact SSE.

    Args:
        event_source: Async generator of event dicts.

    Returns:
        StreamingResponse with ``text/event-stream`` media type.

    """
    return StreamingResponse(
        _sse_generator(event_source),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
