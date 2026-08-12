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

from coro.api.exceptions import UNDECODABLE_MEDIA_MESSAGE, TranscriptionCapacityError
from coro.audio import AudioConversionError
from coro.backends.asr.concurrency import AsrCapacityError
from coro.core.models import PipelineStreamEvent
from coro.pipelines.done_frame import StreamingDoneFrame

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# Sentinel closing every stream, successful or not. Not JSON, so it is the one
# frame no wire dataclass describes; the published AsyncAPI contract derives its
# payload from this constant rather than restating the literal.
SSE_TERMINATOR = "[DONE]"
SSE_MEDIA_TYPE = "text/event-stream"

_TERMINATOR_FRAME = f"data: {SSE_TERMINATOR}\n\n"


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
        yield _TERMINATOR_FRAME
    except AudioConversionError:
        # Client-side problem (unsupported/corrupt media): curated message,
        # invalid_request_error type, and no raw ffmpeg stderr leaked.
        yield _error_frame(UNDECODABLE_MEDIA_MESSAGE, error_type="invalid_request_error")
        yield _TERMINATOR_FRAME
    except AsrCapacityError as exc:
        # Admission control rejected the call. The 200 response headers are
        # already on the wire by the time the generator runs, so the retry hint
        # travels in the error frame's message rather than as Retry-After.
        yield _error_frame(exc.message, error_type=TranscriptionCapacityError.error_type)
        yield _TERMINATOR_FRAME
    except Exception as exc:
        yield _error_frame(str(exc), error_type="server_error")
        yield _TERMINATOR_FRAME


def _error_frame(message: str, *, error_type: str) -> str:
    """Render a single OpenAI-style SSE error frame."""
    payload = json.dumps({"error": {"message": message, "type": error_type}})
    return f"data: {payload}\n\n"


def streaming_response(event_source: AsyncIterator[PipelineStreamEvent]) -> StreamingResponse:
    """Build a StreamingResponse that emits OpenAI-Exact SSE.

    Args:
        event_source: Async generator of ``PipelineStreamEvent`` dataclasses.

    Returns:
        StreamingResponse with ``text/event-stream`` media type.

    """
    return StreamingResponse(
        _sse_generator(event_source),
        media_type=SSE_MEDIA_TYPE,
        headers=_SSE_HEADERS,
    )
