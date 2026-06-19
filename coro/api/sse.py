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

import dataclasses
import json
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from coro.core.models import PipelineStreamEvent
from coro.pipelines.done_frame import StreamingDoneFrame

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _sse_generator(event_source: AsyncIterator[PipelineStreamEvent]):
    r"""Yield SSE-framed lines from an async event source.

    The event source must yield ``PipelineStreamEvent`` dataclasses.
    After all events the generator emits ``data: [DONE]\n\n``.
    On error it emits an error event.
    """
    try:
        async for event in event_source:
            if isinstance(event, StreamingDoneFrame):
                # Rendered straight from the spill store, one row at a time, so
                # the final frame never materialises the whole transcript.
                for line in event.iter_sse():
                    yield line
                continue
            yield f"data: {json.dumps(dataclasses.asdict(event))}\n\n"
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


def streaming_response(event_source: AsyncIterator[PipelineStreamEvent]) -> StreamingResponse:
    """Build a StreamingResponse that emits OpenAI-Exact SSE.

    Args:
        event_source: Async generator of ``PipelineStreamEvent`` dataclasses.

    Returns:
        StreamingResponse with ``text/event-stream`` media type.

    """
    return StreamingResponse(
        _sse_generator(event_source),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
