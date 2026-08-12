"""Boundary Response Schema behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coro.api.schemas import TranscriptionResponse
from coro.api.openai.schemas import OpenAIErrorResponse


def test_transcription_response_rejects_backend_native_extras():
    with pytest.raises(ValidationError):
        TranscriptionResponse.model_validate(
            {
                "segments": [],
                "word_segments": [],
                "transcript": [],
                "diarization": [],
                "raw_words": [],
                "backend_debug": {"native": True},
            }
        )


def test_transcription_response_serializes_public_keys_only():
    response = TranscriptionResponse.model_validate(
        {
            "segments": [],
            "word_segments": [],
            "transcript": [],
            "diarization": [],
            "raw_words": [],
        }
    )

    assert set(response.model_dump()) == {
        "segments",
        "word_segments",
        "transcript",
        "diarization",
        "raw_words",
    }


def test_transcription_response_rejects_extra_fields_inside_items():
    with pytest.raises(ValidationError):
        TranscriptionResponse.model_validate(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "speaker": "1",
                        "words": [],
                        "native_segment": {"leaked": True},
                    }
                ],
                "word_segments": [],
                "transcript": [],
                "diarization": [],
                "raw_words": [],
            }
        )


def test_segment_and_word_overlap_defaults_to_false():
    """The additive overlap flag (ADR 0014) is optional on input."""
    response = TranscriptionResponse.model_validate(
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hola",
                    "speaker": "1",
                    "words": [
                        {"word": "hola", "start": 0.0, "end": 1.0, "score": 1.0, "speaker": "1"}
                    ],
                }
            ],
            "word_segments": [],
            "transcript": [],
            "diarization": [],
            "raw_words": [],
        }
    )

    assert response.segments[0].overlap is False
    assert response.segments[0].words[0].overlap is False


def test_segment_and_word_overlap_round_trips():
    response = TranscriptionResponse.model_validate(
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hola",
                    "speaker": "1",
                    "words": [
                        {
                            "word": "hola",
                            "start": 0.0,
                            "end": 1.0,
                            "score": 1.0,
                            "speaker": "1",
                            "overlap": True,
                        }
                    ],
                    "overlap": True,
                }
            ],
            "word_segments": [],
            "transcript": [],
            "diarization": [],
            "raw_words": [],
        }
    )

    dumped = response.model_dump()["segments"][0]
    assert dumped["overlap"] is True
    assert dumped["words"][0]["overlap"] is True


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
