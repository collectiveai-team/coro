"""Deepgram-native pre-recorded endpoint — POST /v1/listen.

Deepgram defines its own request contract, and this route implements that
contract rather than bending it onto the OpenAI endpoint: a **raw audio body**
(not multipart), Deepgram's query parameters, Deepgram's ``Authorization:
Token`` header, and Deepgram's ``err_code``/``err_msg`` error body.

The route handler stays thin; transcription delegates to the configured
pipeline, exactly as the OpenAI endpoint does. Only the request and response
contracts differ. See ADR 0010.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse

from coro.api.dependencies import get_pipeline, get_settings
from coro.api.schemas import TranscriptionResponse
from coro.api.vendor.deepgram import DeepgramErrorResponse, deepgram_response
from coro.audio import AudioConversionError, AudioInput
from coro.settings import ServerSettings

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

UNDECODABLE_AUDIO_MESSAGE = "Could not decode the submitted audio."
EMPTY_BODY_MESSAGE = "No audio was submitted in the request body."

_BAD_REQUEST = "Bad Request"
_INTERNAL_ERROR = "INTERNAL_SERVER_ERROR"


def _error(
    *, err_code: str, err_msg: str, request_id: str, status_code: int
) -> JSONResponse:
    """Return a Deepgram-shaped error body.

    The app-wide handler emits OpenAI-style ``{"error": {...}}`` objects, which
    a Deepgram client cannot parse, so failures are rendered here instead of
    being raised as ``TranscriptionError``.
    """
    body = DeepgramErrorResponse(err_code=err_code, err_msg=err_msg, request_id=request_id)
    return JSONResponse(body.model_dump(), status_code=status_code)


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


# MARK: Deepgram Pre-Recorded Endpoint
@router.post("/listen", response_model=None)
async def listen(
    request: Request,
    model: str = Query(
        default="", description="Accepted but ignored; server uses configured backend."
    ),
    language: str | None = Query(default=None, description="Optional BCP-47 language hint."),
    diarize: bool = Query(default=False, description="Return per-word speaker labels."),
    utterances: bool = Query(default=False, description="Return the speaker-turn view."),
    punctuate: bool = Query(default=False, description="Accepted but ignored."),
    smart_format: bool = Query(default=False, description="Accepted but ignored."),
    multichannel: bool = Query(default=False, description="Accepted but ignored; audio is mono."),
    numerals: bool = Query(default=False, description="Accepted but ignored."),
    profanity_filter: bool = Query(default=False, description="Accepted but ignored."),
    authorization: str | None = Header(default=None, description="Accepted but not validated."),
    pipeline=Depends(get_pipeline),
    settings: ServerSettings = Depends(get_settings),
) -> JSONResponse:
    """Transcribe a raw audio body and return Deepgram's pre-recorded shape.

    ``diarize`` and ``utterances`` default to ``false``, as they do at
    Deepgram, so per-word speakers require ``?diarize=true&utterances=true``.
    Parameters the configured pipeline cannot honour are accepted and ignored
    rather than rejected, matching how the OpenAI endpoint treats ``model``.
    """
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    audio_bytes = await request.body()
    logger.info(
        "listen[%s] request start bytes=%d content_type=%s diarize=%s utterances=%s language=%s",
        request_id,
        len(audio_bytes),
        request.headers.get("content-type"),
        diarize,
        utterances,
        language,
    )
    if not audio_bytes:
        return _error(
            err_code=_BAD_REQUEST,
            err_msg=EMPTY_BODY_MESSAGE,
            request_id=request_id,
            status_code=400,
        )

    audio = AudioInput(audio_bytes)
    try:
        result = await pipeline.transcribe(audio, language=language, prompt=None)
    except AudioConversionError as exc:
        logger.info("listen[%s] undecodable upload: %s", request_id, exc)
        return _error(
            err_code=_BAD_REQUEST,
            err_msg=UNDECODABLE_AUDIO_MESSAGE,
            request_id=request_id,
            status_code=400,
        )
    except Exception:
        logger.exception(
            "listen[%s] pipeline failed after %.3fs", request_id, time.perf_counter() - started
        )
        return _error(
            err_code=_INTERNAL_ERROR,
            err_msg="Transcription processing failed.",
            request_id=request_id,
            status_code=500,
        )

    validated = TranscriptionResponse.model_validate(asdict(result))
    response = deepgram_response(
        validated,
        text=_text_from_result(validated),
        duration=_duration_from_result(validated),
        request_id=request_id,
        audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
        created=datetime.now(tz=UTC).isoformat(),
        asr_model=settings.model_asr,
        asr_backend=settings.backend_asr,
        diarize=diarize,
        utterances=utterances,
    )
    logger.info(
        "listen[%s] request complete elapsed=%.3fs words=%d",
        request_id,
        time.perf_counter() - started,
        len(validated.word_segments),
    )
    # exclude_none keeps undiarized responses free of null speaker keys, which
    # Deepgram never emits, and drops `utterances` when it was not requested.
    return JSONResponse(response.model_dump(exclude_none=True))
