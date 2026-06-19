"""StreamingDoneFrame renders byte-identical to the materialised SSE frame."""

from __future__ import annotations

import dataclasses
import json

from coro.core.types import SpeakerSegment, TranscriptDoneEvent, TranscriptToken
from coro.pipelines.done_frame import StreamingDoneFrame
from coro.pipelines.finalizer import (
    StreamingTranscriptFinalizer,
    build_streaming_response,
)
from coro.pipelines.transcript_store import TranscriptSpillStore


def _tok(start, end, text, prob=1.0):
    return TranscriptToken(start=start, end=end, text=text, probability=prob)


_TOKENS = [
    _tok(0.0, 0.4, " hola"),
    _tok(0.4, 0.8, ' "mundo".'),  # embedded quotes exercise JSON escaping
    _tok(0.8, 1.2, " cómo"),  # non-ASCII exercises \u escaping
    _tok(1.2, 1.6, " estás?"),
    _tok(1.6, 2.0, " bien"),
    _tok(2.0, 2.4, " gracias."),
]


def _expected_frame(store, timeline) -> str:
    materialized = build_streaming_response(store, timeline)
    event = TranscriptDoneEvent(text=json.dumps(dataclasses.asdict(materialized)))
    return f"data: {json.dumps(dataclasses.asdict(event))}\n\n"


def _populate(store):
    finalizer = StreamingTranscriptFinalizer(store)
    finalizer.add_tokens(_TOKENS)
    finalizer.finish()


def test_streamed_frame_matches_materialized_without_diarization(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    _populate(store)
    expected = _expected_frame(store, [])
    streamed = "".join(StreamingDoneFrame(store=store, timeline=[]).iter_sse())
    assert streamed == expected


def test_streamed_frame_matches_materialized_with_diarization(tmp_path):
    timeline = [
        SpeakerSegment(start=0.0, end=0.8, speaker=2),
        SpeakerSegment(start=0.8, end=2.4, speaker=3),
    ]
    store = TranscriptSpillStore(directory=str(tmp_path))
    _populate(store)
    expected = _expected_frame(store, timeline)
    streamed = "".join(StreamingDoneFrame(store=store, timeline=timeline).iter_sse())
    assert streamed == expected


def test_streamed_frame_inner_json_parses_to_response(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    _populate(store)
    materialized = build_streaming_response(store, [])
    streamed = "".join(StreamingDoneFrame(store=store, timeline=[]).iter_sse())
    outer = json.loads(streamed[len("data: ") :].rstrip("\n"))
    assert json.loads(outer["text"]) == dataclasses.asdict(materialized)


def test_streamed_frame_empty_store(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    expected = _expected_frame(store, [])
    streamed = "".join(StreamingDoneFrame(store=store, timeline=[]).iter_sse())
    assert streamed == expected


def test_iter_sse_closes_store(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    _populate(store)
    path = store.path
    from pathlib import Path

    assert Path(path).exists()
    _ = "".join(StreamingDoneFrame(store=store, timeline=[]).iter_sse())
    assert not Path(path).exists()
