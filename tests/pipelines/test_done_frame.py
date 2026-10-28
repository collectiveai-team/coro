"""StreamingDoneFrame renders byte-identical to the materialised SSE frame."""

from __future__ import annotations

import dataclasses
import json

import pytest

from coro.core.models import SpeakerSegment, TranscriptDoneEvent, TranscriptToken
from coro.pipelines import done_frame
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


def _stream(store, timeline) -> str:
    return "".join(StreamingDoneFrame(store=store, timeline=timeline).iter_sse())


def test_streamed_frame_inner_json_parses_to_response(tmp_path):
    store = TranscriptSpillStore(directory=str(tmp_path))
    _populate(store)
    materialized = build_streaming_response(store, [])
    streamed = _stream(store, [])
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


@dataclasses.dataclass
class _EventDroppingItsText:
    text: str

    def __post_init__(self) -> None:
        self.text = ""


def test_an_event_without_a_text_slot_cannot_frame_the_response():
    with pytest.raises(RuntimeError, match="text slot"):
        done_frame._done_frame_affixes(_EventDroppingItsText)


@dataclasses.dataclass
class _ItemWithAnUnknownField:
    start: float
    confidence_band: str


def test_an_item_field_missing_from_the_segment_is_rejected():
    with pytest.raises(RuntimeError, match=r"\['confidence_band'\]"):
        done_frame._segment_projection(_ItemWithAnUnknownField)


def test_a_response_field_without_a_renderer_is_rejected():
    fields = (*done_frame._RESPONSE_FIELDS, "summary")

    with pytest.raises(RuntimeError, match=r"unrendered fields \['summary'\]"):
        done_frame._check_renderers_cover(fields)


def test_a_renderer_without_a_response_field_is_rejected():
    fields = tuple(name for name in done_frame._RESPONSE_FIELDS if name != "raw_words")

    with pytest.raises(RuntimeError, match=r"unknown renderers \['raw_words'\]"):
        done_frame._check_renderers_cover(fields)
