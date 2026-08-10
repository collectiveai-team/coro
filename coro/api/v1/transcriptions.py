"""Transcription Endpoint router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns OpenAI-shaped JSON
transcription responses. The route handler stays thin; orchestration delegates
to the configured pipeline.
"""

from __future__ import annotations

import re
import hashlib
import math
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import uuid4
from enum import StrEnum
from typing import Literal, overload

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from coro.api.dependencies import get_pipeline, get_settings
from coro.api.exceptions import (
    UNDECODABLE_MEDIA_MESSAGE,
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
from coro.api.vendor import (
    AssemblyAIResponse,
    DeepgramResponse,
    assemblyai_response,
    deepgram_response,
)
from coro.audio import AudioConversionError, AudioInput
from coro.settings import ServerSettings


# MARK: Router Configuration
router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

# Permissive BCP-47 shape: 2-3 letter primary subtag plus optional subtags.
# Guards against junk like Swagger's placeholder "string" reaching the backend.
_BCP47_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def _normalize_optional(value: str | None) -> str | None:
    """Collapse empty or whitespace-only form values to None.

    Swagger's "Try it out" submits empty strings for blanked optional fields;
    treating them as unset keeps the contract forgiving.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _validate_language(language: str | None) -> str | None:
    """Normalize and validate the optional BCP-47 language hint.

    Returns None when unset; raises a 400-mapped validation error for values
    that are not plausible language tags instead of letting the ASR backend
    fail with an opaque 500.
    """
    normalized = _normalize_optional(language)
    if normalized is None:
        return None
    if not _BCP47_LANGUAGE_RE.match(normalized):
        raise TranscriptionValidationError(
            f"Invalid language tag {normalized!r}. Expected a BCP-47 code like 'en' or 'es'.",
            param="language",
        )
    return normalized


# MARK: Response
class ResponseFormat(StrEnum):
    """Every response_format value this server recognises.

    JSON-like formats are implemented; ``json_verbose``/``dirized_json`` are
    typo-tolerant aliases of ``verbose_json``/``diarized_json``. The text output
    formats are recognised so they fail with an OpenAI-style 400 (param
    ``response_format``) rather than a generic validation error.

    ``assemblyai_json`` and ``deepgram_json`` are vendor-shaped formats that
    expose per-word speaker labels, for which the OpenAI formats have no slot.
    They are opt-in because they are large — roughly 7x a ``diarized_json``
    body, since both shapes carry every word twice — and they leave the OpenAI
    projections byte-unchanged (ADR 0010).
    """

    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    JSON_VERBOSE = "json_verbose"
    DIARIZED_JSON = "diarized_json"
    DIRIZED_JSON = "dirized_json"

    # Vendor-shaped formats carrying per-word speakers
    ASSEMBLYAI_JSON = "assemblyai_json"
    DEEPGRAM_JSON = "deepgram_json"

    # Unsupported OpenAI formats (recognised but not implemented → 400)
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


# JSON-like formats this server actually renders (vs. the recognised-but-
# unsupported text outputs above).
_JSON_LIKE_FORMATS = frozenset(
    {
        ResponseFormat.JSON,
        ResponseFormat.VERBOSE_JSON,
        ResponseFormat.JSON_VERBOSE,
        ResponseFormat.DIARIZED_JSON,
        ResponseFormat.DIRIZED_JSON,
        ResponseFormat.ASSEMBLYAI_JSON,
        ResponseFormat.DEEPGRAM_JSON,
    }
)

# Formats whose per-word speaker labels require the vendor request context.
_VENDOR_FORMATS = frozenset(
    {
        ResponseFormat.ASSEMBLYAI_JSON,
        ResponseFormat.DEEPGRAM_JSON,
    }
)


@dataclass(frozen=True)
class VendorContext:
    """Request-scoped values the vendor-shaped projections need.

    The OpenAI projections are pure functions of the transcription result; the
    vendor shapes additionally carry provenance (request id, audio digest,
    model identity), so it is threaded in rather than reached for globally.
    """

    request_id: str
    audio_bytes: bytes
    filename: str
    asr_model: str
    asr_backend: str

    def audio_sha256(self) -> str:
        """Hex SHA-256 of the uploaded audio bytes."""
        return hashlib.sha256(self.audio_bytes).hexdigest()

    def created(self) -> str:
        """ISO 8601 UTC completion timestamp."""
        return datetime.now(tz=UTC).isoformat()


TranscriptionApiResponse = (
    JsonResponse
    | VerboseJsonResponse
    | DiarizadJsonResponse
    | AssemblyAIResponse
    | DeepgramResponse
)
"""Every non-streaming body the Transcription Endpoint can return."""


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


def _verbose_json_response(
    result: TranscriptionResponse, *, language: str | None
) -> VerboseJsonResponse:
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


def _assemblyai_response(
    result: TranscriptionResponse, *, language: str | None, context: VendorContext
) -> AssemblyAIResponse:
    return assemblyai_response(
        result,
        text=_text_from_result(result),
        duration=_duration_from_result(result),
        language=language,
        request_id=context.request_id,
        audio_url=context.filename,
    )


def _deepgram_response(
    result: TranscriptionResponse, *, context: VendorContext
) -> DeepgramResponse:
    return deepgram_response(
        result,
        text=_text_from_result(result),
        duration=_duration_from_result(result),
        request_id=context.request_id,
        audio_sha256=context.audio_sha256(),
        created=context.created(),
        asr_model=context.asr_model,
        asr_backend=context.asr_backend,
    )


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> JsonResponse: ...


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.VERBOSE_JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> VerboseJsonResponse: ...


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.DIARIZED_JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> DiarizadJsonResponse: ...


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.ASSEMBLYAI_JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> AssemblyAIResponse: ...


@overload
def _response_for_format(
    response_format: Literal[ResponseFormat.DEEPGRAM_JSON],
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> DeepgramResponse: ...


@overload
def _response_for_format(
    response_format: ResponseFormat,
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> TranscriptionApiResponse: ...


def _response_for_format(
    response_format: ResponseFormat,
    result: TranscriptionResponse,
    *,
    language: str | None,
    context: VendorContext | None = None,
) -> TranscriptionApiResponse:
    if response_format in _VENDOR_FORMATS and context is None:
        raise TranscriptionProcessingError(
            f"Response format '{response_format}' requires request context."
        )

    match response_format:
        case ResponseFormat.JSON:
            return _json_response(result)
        case ResponseFormat.VERBOSE_JSON | ResponseFormat.JSON_VERBOSE:
            return _verbose_json_response(result, language=language)
        case ResponseFormat.DIARIZED_JSON | ResponseFormat.DIRIZED_JSON:
            return _diarized_json_response(result)
        case ResponseFormat.ASSEMBLYAI_JSON if context is not None:
            return _assemblyai_response(result, language=language, context=context)
        case ResponseFormat.DEEPGRAM_JSON if context is not None:
            return _deepgram_response(result, context=context)

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
    response_format: ResponseFormat = Form(
        default=ResponseFormat.JSON, description="Response format."
    ),
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
    # Typed as UploadFile|str so Swagger's empty-string placeholder is accepted
    # (and ignored) instead of failing UploadFile parsing with a 422.
    known_speaker_references: list[UploadFile | str] | None = File(
        default=None,
        alias="known_speaker_references[]",
        description="Accepted but ignored.",
    ),
    pipeline=Depends(get_pipeline),
    settings: ServerSettings = Depends(get_settings),
) -> Response | TranscriptionApiResponse:
    """Accept audio and return an OpenAI-shaped or vendor-shaped response.

    Supported response formats: json, verbose_json/json_verbose,
    diarized_json/dirized_json (and empty), plus assemblyai_json and
    deepgram_json, which additionally carry per-word speaker labels. Other
    OpenAI text output formats are recognised but not implemented.
    """
    # Request Validation ----------------------------------------------------
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    logger.info(
        "transcription[%s] request start filename=%s content_type=%s "
        "stream=%s response_format=%s language=%s",
        request_id,
        file.filename,
        file.content_type,
        stream,
        response_format,
        language,
    )
    language = _validate_language(language)
    prompt_value = _normalize_optional(prompt)
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
        return streaming_response(stream_method(audio, language=language, prompt=prompt_value))

    # JSON Response ---------------------------------------------------------
    try:
        result = await pipeline.transcribe(audio, language=language, prompt=prompt_value)
    except TranscriptionValidationError:
        raise
    except AudioConversionError as exc:
        logger.info(
            "transcription[%s] undecodable upload after %.3fs: %s",
            request_id,
            time.perf_counter() - started,
            exc,
        )
        raise TranscriptionValidationError(UNDECODABLE_MEDIA_MESSAGE, param="file") from exc
    except Exception as exc:
        logger.exception(
            "transcription[%s] pipeline failed after %.3fs",
            request_id,
            time.perf_counter() - started,
        )
        raise TranscriptionProcessingError("Transcription processing failed.") from exc
    validated = TranscriptionResponse.model_validate(asdict(result))
    logger.info(
        "transcription[%s] request complete elapsed=%.3fs segments=%d words=%d diarization=%d",
        request_id,
        time.perf_counter() - started,
        len(validated.segments),
        len(validated.word_segments or validated.raw_words),
        len(validated.diarization),
    )

    context = VendorContext(
        request_id=request_id,
        audio_bytes=audio_bytes,
        filename=file.filename or "",
        asr_model=settings.model_asr,
        asr_backend=settings.backend_asr,
    )
    return _response_for_format(response_format, validated, language=language, context=context)
