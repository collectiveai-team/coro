"""The internal transcription result is closed, and its field order is the wire order.

These properties used to be enforced per request by validating the pipeline's
result into a pydantic mirror with ``extra="forbid"``. That mirror was 80% of
the audio-proportional heap on the JSON path and is gone (ADR 0018), so the
guarantees it carried are asserted here instead — once, in CI, rather than on
every request in proportion to how long the audio was.

Two things are pinned:

- **Closure.** A backend-native extra cannot reach a public response, because a
  dataclass rejects unknown fields at construction. That is strictly stronger
  than ``extra="forbid"``, which only rejects them when validating from a dict.
- **Field order.** ``StreamingDoneFrame`` derives the done event's JSON key
  order from these declarations, so reordering a field silently reorders
  published output. The order is therefore part of the contract, not an
  implementation detail.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from coro.api.openai.schemas import OpenAIErrorResponse
from coro.core.models import (
    DiarizationItem,
    RawWord,
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)

# The published shape of every type reachable in a transcription response, in
# serialisation order. Spelled out rather than derived so that changing the
# model has to change this list too — a test that reads the answer off the code
# it is testing would pass for any answer.
_PUBLIC_FIELDS = {
    TranscriptionResult: [
        "segments",
        "word_segments",
        "transcript",
        "diarization",
        "raw_words",
        "detected_language",
    ],
    ResponseSegment: ["start", "end", "text", "speaker", "words", "overlap"],
    TranscriptWord: ["word", "start", "end", "score", "speaker", "overlap"],
    TranscriptItem: ["start", "end", "text"],
    DiarizationItem: ["start", "end", "speaker"],
    RawWord: ["word", "start", "end", "score"],
}


@pytest.mark.parametrize(
    ("model", "expected"),
    list(_PUBLIC_FIELDS.items()),
    ids=[model.__name__ for model in _PUBLIC_FIELDS],
)
def test_response_models_declare_exactly_the_public_fields_in_wire_order(model, expected):
    assert is_dataclass(model)
    assert [f.name for f in fields(model)] == expected


@pytest.mark.parametrize(
    "model",
    list(_PUBLIC_FIELDS),
    ids=[model.__name__ for model in _PUBLIC_FIELDS],
)
def test_response_models_reject_backend_native_extras(model):
    """A backend-native field cannot be smuggled into a public response type."""
    with pytest.raises(TypeError):
        model(backend_debug={"native": True})


def test_segment_and_word_overlap_default_to_false():
    """The additive overlap flag (ADR 0014) stays optional to construct."""
    segment = ResponseSegment(
        start=0.0,
        end=1.0,
        text="hola",
        speaker="1",
        words=[TranscriptWord(word="hola", start=0.0, end=1.0, score=1.0, speaker="1")],
    )

    assert segment.overlap is False
    assert segment.words[0].overlap is False


def test_openai_error_response_shape():
    response = OpenAIErrorResponse.from_error(
        message="bad request",
        error_type="invalid_request_error",
        param="file",
    )

    assert response.model_dump() == {
        "error": {
            "message": "bad request",
            "type": "invalid_request_error",
            "param": "file",
            "code": None,
        }
    }
