"""Boundary Response Schema behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coro.api.schemas import OpenAIErrorResponse, TranscriptionResponse


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
