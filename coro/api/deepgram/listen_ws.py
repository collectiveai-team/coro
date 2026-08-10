"""Deepgram-native live endpoint — WebSocket /v1/listen.

Deepgram's streaming contract is a WebSocket, not SSE: the client opens a
socket, declares its audio format in the query string, streams raw samples as
binary frames, and receives ``Results`` frames as they are transcribed,
followed by a closing ``Metadata`` frame. Control is in-band as JSON text
frames (``KeepAlive``, ``Finalize``, ``CloseStream``).

This is genuine live transcription, not a buffer-then-transcribe imitation:
audio flows into the same ``ASRWindowing`` the Streaming Pipeline uses, and
results are emitted as each window completes. See ADR 0010.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from coro.api.deepgram.schemas import DeepgramWord
from coro.audio import SAMPLE_RATE
from coro.api.deepgram.live_schemas import (
    DeepgramLiveError,
    DeepgramLiveMetadata,
    DeepgramLiveModelInfo,
    DeepgramLiveResultsMetadata,
    live_results,
)
from coro.core.models import TranscriptToken
from coro.core.protocols import ASRAdapter
from coro.core.speakers import attribute_span, merge_speaker_timeline
from coro.pcm import PcmStreamConverter, UnsupportedAudioFormat, validate_format
from coro.pipelines.live import LiveAudioSource, LiveTranscriptionSession

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

CLOSE_STREAM = "CloseStream"
FINALIZE = "Finalize"
KEEP_ALIVE = "KeepAlive"

# 1000 Normal Closure; 1008 Policy Violation for a rejected declaration.
_CLOSE_NORMAL = 1000
_CLOSE_POLICY = 1008

UNKNOWN_SPEAKER_LABEL = "-1"


@dataclass(frozen=True)
class _Negotiated:
    """Everything settled at connect time, before any audio is accepted."""

    asr: ASRAdapter
    runtime: Any
    sample_rate: int
    diarize: bool
    language: str | None


def _int_param(websocket: WebSocket, name: str) -> int | None:
    raw = websocket.query_params.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise UnsupportedAudioFormat(f"{name} must be an integer, got {raw!r}") from exc


def _flag(websocket: WebSocket, name: str) -> bool:
    return (websocket.query_params.get(name) or "").strip().lower() in {"true", "1", "yes"}


def _words_from_tokens(tokens: list[TranscriptToken], *, diarize: bool) -> list[DeepgramWord]:
    """Convert accepted ASR tokens into Deepgram live words.

    Interim frames carry no speaker: the diarization timeline is still being
    built while audio arrives, so a label here would be a guess that a later
    frame silently contradicts.
    """
    return [
        DeepgramWord(
            word=token.text.strip(),
            start=round(token.start, 2),
            end=round(token.end, 2),
            confidence=float(token.probability) if token.probability is not None else 1.0,
            speaker=None,
        )
        for token in tokens
        if token.text and token.text.strip()
    ]


def _attributed_words(tokens: list[TranscriptToken], timeline: list) -> list[DeepgramWord]:
    """Convert tokens to words, attaching per-word speakers from the timeline."""
    merged = merge_speaker_timeline(timeline)
    words: list[DeepgramWord] = []
    for token in tokens:
        if not token.text or not token.text.strip():
            continue
        speaker = attribute_span(token.start, token.end, merged).speaker
        words.append(
            DeepgramWord(
                word=token.text.strip(),
                start=round(token.start, 2),
                end=round(token.end, 2),
                confidence=float(token.probability) if token.probability is not None else 1.0,
                speaker=None if speaker == int(UNKNOWN_SPEAKER_LABEL) else speaker,
            )
        )
    return words


async def _send(websocket: WebSocket, model) -> None:
    if websocket.client_state is WebSocketState.CONNECTED:
        await websocket.send_text(model.model_dump_json(exclude_none=True))


async def _reject(websocket: WebSocket, *, description: str, message: str) -> None:
    """Send an ``Error`` frame and close with a policy-violation code."""
    await _send(websocket, DeepgramLiveError(description=description, message=message))
    await websocket.close(code=_CLOSE_POLICY)


async def _negotiate(websocket: WebSocket, request_id: str) -> _Negotiated | None:
    """Validate readiness and the client's declared audio format.

    Returns ``None`` after closing the socket when the connection cannot
    proceed, so a misconfigured client learns at connect time rather than
    after streaming audio that decodes to noise.
    """
    runtime = getattr(websocket.app.state, "runtime", None)
    asr = getattr(runtime, "asr_adapter", None) if runtime else None
    if asr is None:
        await _reject(
            websocket, description="Server not ready", message="No ASR adapter is loaded."
        )
        return None
    try:
        sample_rate = _int_param(websocket, "sample_rate")
        validate_format(
            websocket.query_params.get("encoding"),
            sample_rate,
            _int_param(websocket, "channels"),
        )
    except UnsupportedAudioFormat as exc:
        logger.info("listen_ws[%s] rejected audio declaration: %s", request_id, exc)
        await _reject(websocket, description="Unsupported audio format", message=str(exc))
        return None
    return _Negotiated(
        asr=asr,
        runtime=runtime,
        sample_rate=sample_rate or SAMPLE_RATE,
        diarize=_flag(websocket, "diarize"),
        language=websocket.query_params.get("language") or None,
    )


async def _pump_audio(
    websocket: WebSocket,
    source: LiveAudioSource,
    converter: PcmStreamConverter,
    digest: hashlib._Hash,
) -> None:
    """Forward inbound frames until the client ends the stream.

    The digest accumulates as audio arrives: a live stream has no complete
    payload to hash up front, but ``Metadata`` must still report one.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        if (payload := message.get("bytes")) is not None:
            digest.update(payload)
            await source.push(converter.push(payload))
            continue
        text = message.get("text")
        if text is not None and _control_type(text) in {CLOSE_STREAM, FINALIZE}:
            return
        # KeepAlive and unrecognised control frames hold the socket open.


def _frame_metadata(request_id: str, settings: Any) -> DeepgramLiveResultsMetadata:
    """Model identity carried on every ``Results`` frame, as Deepgram requires."""
    model = getattr(settings, "model_asr", "") if settings else ""
    return DeepgramLiveResultsMetadata(
        request_id=request_id,
        model_uuid=model,
        model_info=DeepgramLiveModelInfo(
            name=model,
            version=getattr(settings, "backend_asr", "") if settings else "",
            arch=getattr(settings, "backend_asr", "") if settings else "",
        ),
    )


# MARK: Deepgram Live Endpoint
@router.websocket("/listen")
async def listen_ws(websocket: WebSocket) -> None:
    """Transcribe a live audio stream and push Deepgram-shaped frames.

    Query parameters mirror Deepgram's: ``encoding`` and ``sample_rate``
    declare the inbound audio, ``diarize`` requests per-word speakers, and
    ``language`` is an optional hint. Unhonoured parameters are ignored, as on
    the REST endpoint.
    """
    await websocket.accept()
    request_id = uuid4().hex[:8]
    negotiated = await _negotiate(websocket, request_id)
    if negotiated is None:
        return

    converter = PcmStreamConverter(source_rate=negotiated.sample_rate)
    source = LiveAudioSource()
    session = LiveTranscriptionSession(
        asr=negotiated.asr,
        streaming_diarizer_factory=(
            getattr(negotiated.runtime, "streaming_diarizer_factory", None)
            if negotiated.diarize
            else None
        ),
        language=negotiated.language,
    )
    logger.info(
        "listen_ws[%s] open diarize=%s sample_rate=%s resampling=%s",
        request_id,
        negotiated.diarize,
        negotiated.sample_rate,
        converter.resampling,
    )

    collected: list[TranscriptToken] = []
    digest = hashlib.sha256()
    settings = getattr(websocket.app.state, "settings", None)
    frame_metadata = _frame_metadata(request_id, settings)

    async def _emit_results() -> None:
        async for tokens in session.run(source):
            collected.extend(tokens)
            await _send(
                websocket,
                live_results(
                    _words_from_tokens(tokens, diarize=negotiated.diarize),
                    start=round(min(token.start for token in tokens), 2),
                    duration=round(
                        max(
                            0.0,
                            max(t.end for t in tokens) - min(t.start for t in tokens),
                        ),
                        2,
                    ),
                    metadata=frame_metadata,
                ),
            )

    consumer = asyncio.create_task(_emit_results())
    try:
        await _pump_audio(websocket, source, converter, digest)
    except WebSocketDisconnect:
        logger.info("listen_ws[%s] client disconnected", request_id)
    finally:
        with contextlib.suppress(Exception):
            await source.push(converter.flush())
        await source.close()
        with contextlib.suppress(Exception):
            await consumer

    await _close_out(
        websocket,
        request_id,
        session,
        collected,
        diarize=negotiated.diarize,
        audio_sha256=digest.hexdigest(),
        frame_metadata=frame_metadata,
    )


async def _close_out(
    websocket: WebSocket,
    request_id: str,
    session: LiveTranscriptionSession,
    collected: list[TranscriptToken],
    *,
    diarize: bool,
    audio_sha256: str,
    frame_metadata: DeepgramLiveResultsMetadata,
) -> None:
    """Emit the attributed final frame (if any) and the closing Metadata."""
    timeline = session.finalize()
    if diarize and timeline and collected:
        await _send(
            websocket,
            live_results(
                _attributed_words(collected, timeline),
                start=0.0,
                duration=round(session.audio_seconds, 2),
                metadata=frame_metadata,
            ),
        )
    settings = getattr(websocket.app.state, "settings", None)
    await _send(
        websocket,
        DeepgramLiveMetadata(
            request_id=request_id,
            sha256=audio_sha256,
            created=datetime.now(tz=UTC).isoformat(),
            duration=round(session.audio_seconds, 2),
            channels=1,
            models=[getattr(settings, "model_asr", "")] if settings else [],
        ),
    )
    if websocket.client_state is WebSocketState.CONNECTED:
        await websocket.close(code=_CLOSE_NORMAL)
    logger.info(
        "listen_ws[%s] closed audio_s=%.2f words=%d",
        request_id,
        session.audio_seconds,
        len(collected),
    )
    logger.info(
        "listen_ws[%s] closed audio_s=%.2f words=%d",
        request_id,
        session.audio_seconds,
        len(collected),
    )


def _control_type(text: str) -> str:
    """Return the ``type`` of a JSON control frame, or '' if unparseable."""
    try:
        return str(json.loads(text).get("type", ""))
    except (ValueError, AttributeError):
        return ""
