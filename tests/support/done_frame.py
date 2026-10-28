"""Rendering a Streaming Pipeline done frame back into its response body."""

from __future__ import annotations

import json
from typing import Any

from coro.pipelines.done_frame import StreamingDoneFrame


def render_done_frame(frame: StreamingDoneFrame) -> Any:
    """Render a StreamingDoneFrame to its SSE bytes and parse the response dict.

    Rendering also closes the frame's spill store.
    """
    sse = "".join(frame.iter_sse())
    assert sse.startswith('data: {"text": "')
    assert sse.endswith('", "type": "transcript.text.done"}\n\n')
    outer = json.loads(sse[len("data: ") :].rstrip("\n"))
    assert outer["type"] == "transcript.text.done"
    return json.loads(outer["text"])
