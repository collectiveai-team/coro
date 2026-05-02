"""Transcription Endpoint router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns OpenAI-shaped JSON
transcription responses. The route handler stays thin; orchestration delegates
to the configured pipeline.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from asr_diar_server.api.dependencies import get_pipeline
from asr_diar_server.api.exceptions import (
    TranscriptionProcessingError,
    TranscriptionValidationError,
    UnsupportedStreamingError,
)
from asr_diar_server.api.schemas import (
    DiarizadJsonResponse,
    DiarizadJsonSegment,
    JsonResponse,
    TranscriptionUsage,
    VerboseJsonResponse,
    VerboseJsonSegment,
    VerboseJsonWord,
    WhisperXResponse,
)
from asr_diar_server.api.sse import streaming_response
from asr_diar_server.audio import AudioInput


# MARK: Router Configuration
router = APIRouter(prefix="/v1")


# MARK: Response
class ResponseFormat(StrEnum):
    """All OpenAI response_format values this server recognises."""

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    DIARIZED_JSON = "diarized_json"

    # Unsupported OpenAI formats (recognised but not implemented)
    # TEXT = "text"
    # SRT = "srt"
    # VTT = "vtt"
    # TSV = "tsv"





def _text_from_result(result: dict[str, Any]) -> str:
    transcript = result.get("transcript") or []
    if transcript:
        return " ".join(str(item.get("text", "")).strip() for item in transcript).strip()
    segments = result.get("segments") or []
    return " ".join(str(item.get("text", "")).strip() for item in segments).strip()


def _duration_from_result(result: dict[str, Any]) -> float:
    ends: list[float] = []
    for key in ("segments", "word_segments", "raw_words", "transcript", "diarization"):
        for item in result.get(key) or []:
            end = item.get("end")
            if isinstance(end, int | float):
                ends.append(float(end))
    return max(ends, default=0.0)


def _usage(duration: float) -> TranscriptionUsage:
    return TranscriptionUsage(type="duration", seconds=math.ceil(duration))


def _json_response(result: dict[str, Any]) -> JsonResponse:
    duration = _duration_from_result(result)
    return JsonResponse(text=_text_from_result(result), usage=_usage(duration))


def _verbose_json_response(result: dict[str, Any], *, language: str | None) -> VerboseJsonResponse:
    duration = _duration_from_result(result)
    return VerboseJsonResponse(
        duration=duration,
        language=language or "unknown",
        text=_text_from_result(result),
        segments=[
            VerboseJsonSegment(
                id=index,
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                text=segment.get("text", ""),
                tokens=[],
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
            )
            for index, segment in enumerate(result.get("segments") or [])
        ],
        words=[
            VerboseJsonWord(
                word=word.get("word", ""),
                start=word.get("start", 0.0),
                end=word.get("end", 0.0),
            )
            for word in result.get("word_segments") or result.get("raw_words") or []
        ],
        usage=_usage(duration),
    )


def _diarized_json_response(result: dict[str, Any]) -> DiarizadJsonResponse:
    duration = _duration_from_result(result)
    return DiarizadJsonResponse(
        task="transcribe",
        duration=duration,
        text=_text_from_result(result),
        segments=[
            DiarizadJsonSegment(
                type="transcript.text.segment",
                id=f"seg_{index + 1:03d}",
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                text=segment.get("text", ""),
                speaker=segment.get("speaker", "unknown"),
            )
            for index, segment in enumerate(result.get("segments") or [])
        ],
        usage=_usage(duration),
    )


# MARK: Transcription Endpoint
@router.post("/audio/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(
        default="", description="Accepted but ignored; server uses configured backend."
    ),
    language: str | None = Form(default=None, description="Optional BCP-47 language hint."),
    prompt: str = Form(default="", description="Optional initial prompt for transcription."),
    response_format: ResponseFormat = Form(default=ResponseFormat.JSON, description="Response format."),
    temperature: float | None = Form(default=None, description="Accepted but ignored."),
    timestamp_granularities: list[str] | None = Form(
        default=None,
        alias="timestamp_granularities[]",
        description="Accepted but ignored.",
    ),
    stream: bool = Form(default=False, description="If true, return OpenAI-Exact SSE."),
    include: list[str] | None = Form(
        default=None,
        alias="include[]",
        description="Accepted but ignored.",
    ),
    chunking_strategy: str | None = Form(default=None, description="Accepted but ignored."),
    known_speaker_names: list[str] | None = Form(
        default=None,
        alias="known_speaker_names[]",
        description="Accepted but ignored.",
    ),
    known_speaker_references: list[UploadFile] | None = File(
        default=None,
        alias="known_speaker_references[]",
        description="Accepted but ignored.",
    ),
    pipeline=Depends(get_pipeline),
) -> Response | JsonResponse | VerboseJsonResponse | DiarizadJsonResponse:
    """Accept audio and return an OpenAI-shaped response.

    Supported response formats: json, verbose_json/json_verbose,
    diarized_json/dirized_json (and empty). Other OpenAI text output formats
    are recognised but not implemented.
    """
    # Request Validation ----------------------------------------------------
    audio = await AudioInput.from_upload(file)
    if not await audio.read_bytes():
        raise TranscriptionValidationError("Empty audio file.", param="file")

    # Streaming Response ----------------------------------------------------
    if stream:
        stream_method = getattr(pipeline, "stream", None)
        if stream_method is None:
            raise UnsupportedStreamingError("Configured pipeline does not support streaming.")
        return streaming_response(stream_method(audio, language=language, prompt=prompt or None))

    # JSON Response ---------------------------------------------------------
    try:
        result = await pipeline.transcribe(audio, language=language, prompt=prompt or None)
    except TranscriptionValidationError:
        raise
    except Exception as exc:
        raise TranscriptionProcessingError("Transcription processing failed.") from exc
    validated = WhisperXResponse.model_validate(result).model_dump()

    match response_format:
        case ResponseFormat.JSON:
            return _json_response(validated)
        case ResponseFormat.VERBOSE_JSON:
            return _verbose_json_response(validated, language=language)
        case ResponseFormat.DIARIZED_JSON:
            return _diarized_json_response(validated)

        case _:
            raise TranscriptionValidationError(
                f"Unsupported response_format '{response_format}'.",
                param="response_format",
            )
