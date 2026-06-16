"""Transcription Endpoint router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns OpenAI-shaped JSON
transcription responses. The route handler stays thin; orchestration delegates
to the configured pipeline.
"""

from __future__ import annotations

import math
import logging
import time
from uuid import uuid4
from enum import StrEnum
from typing import Literal, overload

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from coro.api.dependencies import get_pipeline
from coro.api.exceptions import (
    TranscriptionProcessingError,
    TranscriptionValidationError,
    UnsupportedStreamingError,
)
from coro.api.schemas import (
    DiarizadJsonResponse,
    DiarizadJsonSegment,
    JsonResponse,
    TranscriptionResponse,
    TranscriptionUsage,
    VerboseJsonResponse,
    VerboseJsonSegment,
    VerboseJsonWord,
)
from coro.api.sse import streaming_response
from coro.audio import AudioInput


# MARK: Router Configuration
router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


# MARK: Response
class ResponseFormat(StrEnum):
    """All OpenAI response_format values this server recognises.

    JSON-like formats are implemented; ``json_verbose``/``dirized_json`` are
    typo-tolerant aliases of ``verbose_json``/``diarized_json``. The text output
    formats are recognised so they fail with an OpenAI-style 400 (param
    ``response_format``) rather than a generic validation error.
    """

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    JSON_VERBOSE = "json_verbose"
    DIARIZED_JSON = "diarized_json"
    DIRIZED_JSON = "dirized_json"

    # Unsupported OpenAI formats (recognised but not implemented → 400)
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


# JSON-like formats this server actually renders (vs. the recognised-but-
# unsupported text outputs above).
_JSON_LIKE_FORMATS = frozenset({
    ResponseFormat.JSON,
    ResponseFormat.VERBOSE_JSON,
    ResponseFormat.JSON_VERBOSE,
    ResponseFormat.DIARIZED_JSON,
    ResponseFormat.DIRIZED_JSON,
})


def _text_from_result(result: TranscriptionResponse) -> str:
    if result.transcript:
        return " ".join(item.text.strip() for item in result.transcript).strip()
    return " ".join(segment.text.strip() for segment in result.segments).strip()


def _duration_from_result(result: TranscriptionResponse) -> float:
    return max(
        [
            item.end
            for items in (
                result.segments,
                result.word_segments,
                result.raw_words,
                result.transcript,
                result.diarization,
            )
            for item in items
        ],
        default=0.0,
    )


def _usage(duration: float) -> TranscriptionUsage:
    return TranscriptionUsage(type="duration", seconds=math.ceil(duration))


def _json_response(result: TranscriptionResponse) -> JsonResponse:
    duration = _duration_from_result(result)
    return JsonResponse(text=_text_from_result(result), usage=_usage(duration))


def _verbose_json_response(result: TranscriptionResponse, *, language: str | None) -> VerboseJsonResponse:
    duration = _duration_from_result(result)
    return VerboseJsonResponse(
        duration=duration,
        language=language or "unknown",
        text=_text_from_result(result),
        segments=[
            VerboseJsonSegment(
                id=index,
                seek=int(segment.start * 100),
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=[],
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
            )
            for index, segment in enumerate(result.segments)
        ],
        words=[
            VerboseJsonWord(
                word=word.word,
                start=word.start,
                end=word.end,
            )
            for word in result.word_segments or result.raw_words
        ],
        usage=_usage(duration),
    )


def _diarized_json_response(result: TranscriptionResponse) -> DiarizadJsonResponse:
    duration = _duration_from_result(result)
    return DiarizadJsonResponse(
        task="transcribe",
        duration=duration,
        text=_text_from_result(result),
        segments=[
            DiarizadJsonSegment(
                type="transcript.text.segment",
                id=f"seg_{index + 1:03d}",
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=segment.speaker,
            )
            for index, segment in enumerate(result.segments)
        ],
        usage=_usage(duration),
    )


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
) -> JsonResponse: ...


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.VERBOSE_JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
) -> VerboseJsonResponse: ...


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.DIARIZED_JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
) -> DiarizadJsonResponse: ...


def _response_for_format(
    response_format: ResponseFormat,
    result: TranscriptionResponse,
    *,
    language: str | None,
) -> JsonResponse | VerboseJsonResponse | DiarizadJsonResponse:
    match response_format:
        case ResponseFormat.JSON:
            return _json_response(result)
        case ResponseFormat.VERBOSE_JSON | ResponseFormat.JSON_VERBOSE:
            return _verbose_json_response(result, language=language)
        case ResponseFormat.DIARIZED_JSON | ResponseFormat.DIRIZED_JSON:
            return _diarized_json_response(result)

    raise TranscriptionValidationError(
        f"Unsupported response_format '{response_format}'.",
        param="response_format",
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
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    logger.info(
        "transcription[%s] request start filename=%s content_type=%s stream=%s response_format=%s language=%s",
        request_id,
        file.filename,
        file.content_type,
        stream,
        response_format,
        language,
    )
    audio = await AudioInput.from_upload(file)
    audio_bytes = await audio.read_bytes()
    logger.info("transcription[%s] upload read bytes=%d", request_id, len(audio_bytes))
    if not audio_bytes:
        raise TranscriptionValidationError("Empty audio file.", param="file")

    # Streaming Response ----------------------------------------------------
    if stream:
        stream_method = getattr(pipeline, "stream", None)
        if stream_method is None:
            raise UnsupportedStreamingError("Configured pipeline does not support streaming.")
        logger.info("transcription[%s] handing off to streaming response", request_id)
        return streaming_response(stream_method(audio, language=language, prompt=prompt or None))

    # JSON Response ---------------------------------------------------------
    try:
        result = await pipeline.transcribe(audio, language=language, prompt=prompt or None)
    except TranscriptionValidationError:
        raise
    except Exception as exc:
        logger.exception("transcription[%s] pipeline failed after %.3fs", request_id, time.perf_counter() - started)
        raise TranscriptionProcessingError("Transcription processing failed.") from exc
    validated = TranscriptionResponse.model_validate(result)
    logger.info(
        "transcription[%s] request complete elapsed=%.3fs segments=%d words=%d diarization=%d",
        request_id,
        time.perf_counter() - started,
        len(validated.segments),
        len(validated.word_segments or validated.raw_words),
        len(validated.diarization),
    )

    return _response_for_format(response_format, validated, language=language)
