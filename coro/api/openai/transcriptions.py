"""Transcription Endpoint router — /v1/audio/transcriptions.

Accepts OpenAI-compatible form parameters and returns OpenAI-shaped JSON
transcription responses. The route handler stays thin; orchestration delegates
to the configured pipeline and rendering to the incremental renderers.
"""

from __future__ import annotations

import re
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from coro.api.dependencies import get_pipeline
from coro.api.exceptions import (
    UNDECODABLE_MEDIA_MESSAGE,
    TranscriptionCapacityError,
    TranscriptionProcessingError,
    TranscriptionValidationError,
    UnsupportedStreamingError,
)
from coro.api.json_body import spooled_json_response
from coro.api.openai.formats import ResponseFormat
from coro.api.openai.render import render_for_format
from coro.api.openai.sse import streaming_response
from coro.audio import AudioConversionError, AudioInput
from coro.backends.asr.concurrency import AsrCapacityError
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.pipelines.source import transcript_source


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


async def _transcribe_or_raise(
    pipeline,
    audio: AudioInput,
    *,
    language: str | None,
    prompt: str | None,
    request_id: str,
    started: float,
):
    """Call the pipeline and translate its typed failures into ``TranscriptionError``.

    Extracted from :func:`create_transcription` so each backend failure mode
    (capacity, unsupported language, undecodable media, anything else) is one
    branch here rather than inflating that route handler's complexity.
    """
    try:
        return await transcript_source(pipeline, audio, language=language, prompt=prompt)
    except TranscriptionValidationError:
        raise
    except AsrCapacityError as exc:
        # Admission control rejected the call: shed load with a retry hint rather
        # than reporting it as a server fault.
        logger.info(
            "transcription[%s] rejected at ASR capacity after %.3fs: %s",
            request_id,
            time.perf_counter() - started,
            exc,
        )
        raise TranscriptionCapacityError(
            exc.message, retry_after_seconds=exc.retry_after_seconds
        ) from exc
    except AsrUnsupportedLanguageError as exc:
        logger.info(
            "transcription[%s] rejected unsupported language after %.3fs: %s",
            request_id,
            time.perf_counter() - started,
            exc,
        )
        raise TranscriptionValidationError(exc.message, param="language") from exc
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
) -> Response:
    """Accept audio and return an OpenAI-shaped response.

    Supported response formats: json, verbose_json and diarized_json (and
    empty). Other OpenAI text output formats are recognised but not
    implemented.

    The body is rendered incrementally from a Transcript Source and spooled to
    disk, so it is served with a real ``Content-Length`` without ever being
    fully resident (ADR 0018).
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
    # Size comes from the spool counter, never from reading the upload back: a
    # multi-gigabyte upload must not be materialised just to be measured.
    logger.info("transcription[%s] upload spooled bytes=%d", request_id, audio.size)
    if not audio.size:
        await audio.cleanup()
        raise TranscriptionValidationError("Empty audio file.", param="file")

    # Streaming Response ----------------------------------------------------
    if stream:
        stream_method = getattr(pipeline, "stream", None)
        if stream_method is None:
            await audio.cleanup()
            raise UnsupportedStreamingError("Configured pipeline does not support streaming.")
        logger.info("transcription[%s] handing off to streaming response", request_id)
        return streaming_response(stream_method(audio, language=language, prompt=prompt_value))

    # JSON Response ---------------------------------------------------------
    source = await _transcribe_or_raise(
        pipeline,
        audio,
        language=language,
        prompt=prompt_value,
        request_id=request_id,
        started=started,
    )

    # The body is rendered before the response exists, so a projection failure is
    # still a 500 with an OpenAI-Style Error rather than a truncated 200.
    try:
        response = spooled_json_response(
            render_for_format(response_format, source, language=language)
        )
    except TranscriptionValidationError:
        raise
    except Exception as exc:
        logger.exception("transcription[%s] response rendering failed", request_id)
        raise TranscriptionProcessingError("Transcription processing failed.") from exc
    finally:
        source.close()

    logger.info(
        "transcription[%s] request complete elapsed=%.3fs format=%s body_bytes=%s",
        request_id,
        time.perf_counter() - started,
        response_format,
        response.headers.get("content-length"),
    )
    return response
