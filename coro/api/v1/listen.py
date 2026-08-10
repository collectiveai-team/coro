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
from collections.abc import Mapping
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
URL_INGEST_MESSAGE = "Remote URL ingest is not supported. Submit the audio as the raw request body."

_BAD_REQUEST = "Bad Request"
_INTERNAL_ERROR = "INTERNAL_SERVER_ERROR"

# Parameters that change the *contract* rather than transcription quality:
# they alter the interaction model, add response keys coro cannot produce, or
# carry a compliance guarantee. Accepting these silently would return a body
# that does not answer the request the client actually made, so they are
# refused. Quality-only knobs (punctuate, numerals, smart_format, ...) stay
# accepted-and-ignored. See ADR 0010.
_UNSUPPORTED_PARAMS: dict[str, str] = {
    "callback": "asynchronous callback delivery is not supported",
    "callback_method": "asynchronous callback delivery is not supported",
    "summarize": "summarization is not supported",
    "sentiment": "sentiment analysis is not supported",
    "topics": "topic detection is not supported",
    "custom_topic": "topic detection is not supported",
    "intents": "intent recognition is not supported",
    "custom_intent": "intent recognition is not supported",
    "detect_entities": "entity detection is not supported",
    "paragraphs": "paragraph segmentation is not supported",
    "search": "term search is not supported",
    "measurements": "measurement formatting is not supported",
    "redact": "PII redaction is not supported",
    "replace": "term replacement is not supported",
    "detect_language": "language detection is not supported",
}

_DISABLED_VALUES = {"false", "0", "no"}


def _is_requested(value: str) -> bool:
    """Return True when a query value asks for the feature.

    Anything that is not an explicit disable counts, because several of these
    parameters carry a payload rather than a boolean — ``callback`` takes a
    URL, ``search`` a term, ``redact`` a policy, and ``summarize`` accepts
    ``v2`` as well as ``true``. A bare flag with no value is a request too.
    """
    return value.strip().lower() not in _DISABLED_VALUES


def _unsupported_parameter(params: Mapping[str, str]) -> tuple[str, str] | None:
    """Return the first requested parameter coro cannot honour, if any.

    Deepgram's defaults are all off, so ``summarize=false`` asks for nothing
    and is not refused.
    """
    for name, reason in _UNSUPPORTED_PARAMS.items():
        value = params.get(name)
        if value is not None and _is_requested(value):
            return name, reason
    # multichannel is separate: audio is downmixed to mono, so honouring it
    # would silently return one channel where the client expects several.
    multichannel = params.get("multichannel")
    if multichannel is not None and _is_requested(multichannel):
        return "multichannel", "audio is converted to mono; multichannel is not supported"
    return None


def _error(*, err_code: str, err_msg: str, request_id: str, status_code: int) -> JSONResponse:
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

    Quality-only parameters coro cannot honour (``punctuate``, ``numerals``,
    ``smart_format``, ...) are accepted and ignored. Parameters that would
    change the response contract — asynchronous ``callback`` delivery, the
    analysis features, ``redact``, ``multichannel`` — are refused instead, so a
    client never receives a body that silently fails to answer its request.
    """
    request_id = uuid4().hex[:8]
    started = time.perf_counter()

    unsupported = _unsupported_parameter(request.query_params)
    if unsupported is not None:
        name, reason = unsupported
        logger.info("listen[%s] refused unsupported parameter %s", request_id, name)
        return _error(
            err_code=_BAD_REQUEST,
            err_msg=f"Unsupported parameter '{name}': {reason}.",
            request_id=request_id,
            status_code=400,
        )

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    if content_type == "application/json":
        # Deepgram's URL-ingest mode. Refused explicitly rather than handed to
        # the decoder, which would fail with a misleading "undecodable audio".
        return _error(
            err_code=_BAD_REQUEST,
            err_msg=URL_INGEST_MESSAGE,
            request_id=request_id,
            status_code=400,
        )

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
