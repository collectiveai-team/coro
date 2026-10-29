"""StreamingDoneFrame renders byte-identical to the materialised SSE frame."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path

import pytest

from coro.core import models
from coro.core.models import SpeakerSegment, TranscriptDoneEvent, TranscriptToken
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
    assert Path(path).exists()
    _ = "".join(StreamingDoneFrame(store=store, timeline=[]).iter_sse())
    assert not Path(path).exists()


def _import_done_frame_with(monkeypatch, **model_types):
    """Import a fresh ``coro.pipelines.done_frame`` against substituted response models.

    The module checks its renderers against the models when it is imported,
    which is the only time production code meets that check.
    """
    for name, model_type in model_types.items():
        monkeypatch.setattr(models, name, model_type)
    monkeypatch.delitem(sys.modules, "coro.pipelines.done_frame")
    return importlib.import_module("coro.pipelines.done_frame")


@dataclasses.dataclass
class _EventDroppingItsText:
    text: str

    def __post_init__(self) -> None:
        self.text = ""


def test_an_event_without_a_text_slot_cannot_frame_the_response(monkeypatch):
    with pytest.raises(RuntimeError, match="text slot"):
        _import_done_frame_with(monkeypatch, TranscriptDoneEvent=_EventDroppingItsText)


@dataclasses.dataclass
class _ItemWithAnUnknownField:
    start: float
    confidence_band: str


def test_an_item_field_missing_from_the_segment_is_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match=r"\['confidence_band'\]"):
        _import_done_frame_with(monkeypatch, TranscriptItem=_ItemWithAnUnknownField)


_RESPONSE_FIELD_NAMES = [field.name for field in dataclasses.fields(models.TranscriptionResult)]


def test_a_response_field_without_a_renderer_is_rejected(monkeypatch):
    response = dataclasses.make_dataclass(
        "TranscriptionResult", [*_RESPONSE_FIELD_NAMES, "summary"]
    )

    with pytest.raises(RuntimeError, match=r"unrendered fields \['summary'\]"):
        _import_done_frame_with(monkeypatch, TranscriptionResult=response)


def test_a_renderer_without_a_response_field_is_rejected(monkeypatch):
    response = dataclasses.make_dataclass(
        "TranscriptionResult", [name for name in _RESPONSE_FIELD_NAMES if name != "raw_words"]
    )

    with pytest.raises(RuntimeError, match=r"unknown renderers \['raw_words'\]"):
        _import_done_frame_with(monkeypatch, TranscriptionResult=response)


def test_a_fresh_import_against_the_real_models_still_frames_the_response(monkeypatch, tmp_path):
    module = _import_done_frame_with(monkeypatch)
    store = TranscriptSpillStore(directory=str(tmp_path))
    _populate(store)
    expected = _expected_frame(store, [])

    streamed = "".join(module.StreamingDoneFrame(store=store, timeline=[]).iter_sse())

    assert streamed == expected
