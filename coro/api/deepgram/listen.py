"""Deepgram-native pre-recorded endpoint — POST /v1/listen.

Deepgram defines its own request contract, and this route implements that
contract rather than bending it onto the OpenAI endpoint: a **raw audio body**
(not multipart), Deepgram's query parameters, Deepgram's ``Authorization:
Token`` header, and Deepgram's ``err_code``/``err_msg`` error body.

The route handler stays thin; transcription delegates to the configured
pipeline, exactly as the OpenAI endpoint does. Only the request and response
contracts differ. See ADR 0015.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse, Response

from coro.api.dependencies import get_pipeline, get_settings
from coro.api.deepgram.render import render_deepgram
from coro.api.deepgram.schemas import DeepgramErrorResponse
from coro.api.json_body import spooled_json_response
from coro.audio import AudioConversionError, AudioInput
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.core.transcript_source import TranscriptSource
from coro.pipelines.source import transcript_source
from coro.settings import ServerSettings

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

UNDECODABLE_AUDIO_MESSAGE = "Could not decode the submitted audio."
EMPTY_BODY_MESSAGE = "No audio was submitted in the request body."
URL_INGEST_MESSAGE = "Remote URL ingest is not supported. Submit the audio as the raw request body."

_BAD_REQUEST = "Bad Request"
_INTERNAL_ERROR = "INTERNAL_SERVER_ERROR"

_IGNORED_DOC = "Accepted but ignored; the configured backend does not expose this control."
_NO_FEATURE_DOC = (
    "Accepted but ignored. coro does not compute this, so the corresponding response "
    "key is absent rather than empty."
)

# Almost every unhonoured parameter is accepted and ignored, so a client's
# standard parameter bundle still works and a future Deepgram flag does not
# break the endpoint. Features coro cannot compute simply produce no key, which
# a client reads as absent — recoverable, and documented in the OpenAPI schema
# rather than enforced at runtime.
#
# Two are refused, because ignoring them fails *silently and harmfully* rather
# than producing missing data:
#
# - ``redact`` promises PII was removed. Returning 200 without redacting is a
#   compliance failure wearing a success code; the client cannot detect it.
# - ``callback`` promises delivery to a webhook. The client is built to receive
#   ``{request_id}`` and then wait, so ignoring it hangs that workflow forever
#   rather than handing back data it can inspect.
#
# See ADR 0015.
_UNSUPPORTED_PARAMS: dict[str, str] = {
    "callback": "asynchronous callback delivery is not supported; results are returned inline",
    "callback_method": "asynchronous callback delivery is not supported",
    "redact": "PII redaction is not performed; refusing rather than returning unredacted audio",
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
    """Return the first requested parameter coro must refuse, if any.

    Deepgram's defaults are all off, so ``redact=false`` asks for nothing and
    is not refused.
    """
    for name, reason in _UNSUPPORTED_PARAMS.items():
        value = params.get(name)
        if value is not None and _is_requested(value):
            return name, reason
    return None


def _error(*, err_code: str, err_msg: str, request_id: str, status_code: int) -> JSONResponse:
    """Return a Deepgram-shaped error body.

    The app-wide handler emits OpenAI-style ``{"error": {...}}`` objects, which
    a Deepgram client cannot parse, so failures are rendered here instead of
    being raised as ``TranscriptionError``.
    """
    body = DeepgramErrorResponse(err_code=err_code, err_msg=err_msg, request_id=request_id)
    return JSONResponse(body.model_dump(), status_code=status_code)


def _word_count(source: TranscriptSource) -> int:
    """Count the per-word entries for the completion log, without holding them."""
    return sum(1 for _ in source.iter_words())


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
    punctuate: bool = Query(default=False, description=_IGNORED_DOC),
    smart_format: bool = Query(default=False, description=_IGNORED_DOC),
    numerals: bool = Query(default=False, description=_IGNORED_DOC),
    profanity_filter: bool = Query(default=False, description=_IGNORED_DOC),
    filler_words: bool = Query(default=False, description=_IGNORED_DOC),
    dictation: bool = Query(default=False, description=_IGNORED_DOC),
    multichannel: bool = Query(
        default=False,
        description="Accepted but ignored. Audio is downmixed to mono, so one channel is returned.",
    ),
    detect_language: bool = Query(
        default=False,
        description="Accepted but ignored. `detected_language` is not returned; pass `language`.",
    ),
    summarize: str | None = Query(default=None, description=_NO_FEATURE_DOC),
    sentiment: bool = Query(default=False, description=_NO_FEATURE_DOC),
    topics: bool = Query(default=False, description=_NO_FEATURE_DOC),
    intents: bool = Query(default=False, description=_NO_FEATURE_DOC),
    detect_entities: bool = Query(default=False, description=_NO_FEATURE_DOC),
    paragraphs: bool = Query(default=False, description=_NO_FEATURE_DOC),
    search: list[str] | None = Query(default=None, description=_NO_FEATURE_DOC),
    measurements: bool = Query(default=False, description=_NO_FEATURE_DOC),
    replace: list[str] | None = Query(default=None, description=_NO_FEATURE_DOC),
    redact: list[str] | None = Query(
        default=None,
        description="**Refused with 400.** No redaction is performed, and returning "
        "unredacted text under a redaction request would be a silent compliance failure.",
    ),
    callback: str | None = Query(
        default=None,
        description="**Refused with 400.** Results are returned inline; no webhook is ever "
        "delivered, so a client awaiting one would wait forever.",
    ),
    authorization: str | None = Header(default=None, description="Accepted but not validated."),
    pipeline=Depends(get_pipeline),
    settings: ServerSettings = Depends(get_settings),
) -> Response:
    """Transcribe a raw audio body and return Deepgram's pre-recorded shape.

    ``diarize`` and ``utterances`` default to ``false``, as they do at
    Deepgram, so per-word speakers require ``?diarize=true&utterances=true``.

    Unhonoured parameters are accepted and ignored, so a client's standard
    parameter bundle still works and a future Deepgram flag does not break the
    endpoint. Each is documented above with what its absence means.

    ``redact`` and ``callback`` are the exceptions and are refused with a 400:
    ignoring them fails silently and harmfully rather than merely omitting a
    response key. See ADR 0015.
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
        source = await transcript_source(pipeline, audio, language=language, prompt=None)
    except AudioConversionError as exc:
        logger.info("listen[%s] undecodable upload: %s", request_id, exc)
        return _error(
            err_code=_BAD_REQUEST,
            err_msg=UNDECODABLE_AUDIO_MESSAGE,
            request_id=request_id,
            status_code=400,
        )
    except AsrUnsupportedLanguageError as exc:
        logger.info("listen[%s] rejected unsupported language: %s", request_id, exc)
        return _error(
            err_code=_BAD_REQUEST,
            err_msg=exc.message,
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

    # The body is rendered before the response exists, so a projection failure is
    # still a Deepgram-shaped 500 rather than a truncated 200. exclude_none keeps
    # undiarized responses free of null speaker keys, which Deepgram never emits,
    # and drops `utterances` when it was not requested.
    try:
        words = _word_count(source)
        response = spooled_json_response(
            render_deepgram(
                source,
                request_id=request_id,
                audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
                created=datetime.now(tz=UTC).isoformat(),
                asr_model=settings.model_asr,
                asr_backend=settings.backend_asr,
                diarize=diarize,
                utterances=utterances,
            )
        )
    except Exception:
        logger.exception("listen[%s] response rendering failed", request_id)
        return _error(
            err_code=_INTERNAL_ERROR,
            err_msg="Transcription processing failed.",
            request_id=request_id,
            status_code=500,
        )
    finally:
        source.close()

    logger.info(
        "listen[%s] request complete elapsed=%.3fs words=%d",
        request_id,
        time.perf_counter() - started,
        words,
    )
    return response
