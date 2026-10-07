"""Deepgram-native live endpoint — WebSocket /v1/listen.

Deepgram's streaming contract is a WebSocket, not SSE: the client opens a
socket, declares its audio format in the query string, streams raw samples as
binary frames, and receives ``Results`` frames as they are transcribed,
followed by a closing ``Metadata`` frame. Control is in-band as JSON text
frames (``KeepAlive``, ``Finalize``, ``CloseStream``).

This is genuine live transcription, not a buffer-then-transcribe imitation:
audio flows into the same ``ASRWindowing`` the Streaming Pipeline uses, and
results are emitted as each window completes. See ADR 0015.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket

from coro.api.deepgram.live_socket import (
    CLOSE_STREAM,
    FINALIZE,
    KEEP_ALIVE,
    deny_if_rate_limited,
    read_socket,
    reject,
    send_frame,
    socket_open,
    spill_dir,
)
from coro.api.deepgram.schemas import DeepgramWord
from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.api.deepgram.live_schemas import (
    DeepgramLiveMetadata,
    DeepgramLiveModelInfo,
    DeepgramLiveResultsMetadata,
    live_results,
)
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.core.language import canonical_language
from coro.cache.adapter import unwrap_asr_adapter
from coro.core.models import TranscriptToken
from coro.core.protocols import ASRAdapter
from coro.core.speakers import attribute_span, merge_speaker_timeline
from coro.pcm import PcmStreamConverter, UnsupportedAudioFormat, validate_format
from coro.pipelines.live import LiveAudioSource, LiveTranscriptionSession

__all__ = ["CLOSE_STREAM", "FINALIZE", "KEEP_ALIVE", "router"]

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

# 1000 Normal Closure; rejections close with live_socket.CLOSE_POLICY (1008).
_CLOSE_NORMAL = 1000

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


async def _negotiate(websocket: WebSocket, request_id: str) -> _Negotiated | None:
    """Validate readiness and the client's declared audio format.

    Returns ``None`` after closing the socket when the connection cannot
    proceed, so a misconfigured client learns at connect time rather than
    after streaming audio that decodes to noise.
    """
    runtime = getattr(websocket.app.state, "runtime", None)
    asr = getattr(runtime, "asr_adapter", None) if runtime else None
    if asr is None:
        await reject(websocket, description="Server not ready", message="No ASR adapter is loaded.")
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
        await reject(websocket, description="Unsupported audio format", message=str(exc))
        return None
    language = canonical_language(websocket.query_params.get("language"))
    # Backends that resolve/validate a language up front (currently only
    # onnx-canary-split) get it checked at negotiate time, mirroring the audio
    # format check above -- an unsupported language is rejected before any
    # audio flows rather than surfacing later inside the streaming session.
    resolve_language = getattr(unwrap_asr_adapter(asr), "resolve_language", None)
    if resolve_language is not None:
        try:
            resolve_language(language)
        except AsrUnsupportedLanguageError as exc:
            logger.info("listen_ws[%s] rejected unsupported language: %s", request_id, exc)
            await reject(websocket, description="Unsupported language", message=exc.message)
            return None
    return _Negotiated(
        asr=asr,
        runtime=runtime,
        sample_rate=sample_rate or SAMPLE_RATE,
        diarize=_flag(websocket, "diarize"),
        language=language,
    )


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
    if await deny_if_rate_limited(websocket):
        return
    await websocket.accept()
    request_id = uuid4().hex[:8]
    negotiated = await _negotiate(websocket, request_id)
    if negotiated is None:
        return

    converter = PcmStreamConverter(source_rate=negotiated.sample_rate)
    source = LiveAudioSource(spill_dir=spill_dir(websocket))
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
            await send_frame(
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
    reader = asyncio.create_task(read_socket(websocket, source, converter, digest, request_id))
    try:
        await asyncio.wait({consumer, reader}, return_when=asyncio.FIRST_COMPLETED)
        if reader.done() or not socket_open(websocket):
            # The client is gone, or was closed out for its rate limit: nobody
            # can receive the rest, so the backlog is abandoned rather than
            # burning CPU other connections need.
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await consumer
            logger.info(
                "listen_ws[%s] stream abandoned; backlog_s=%.2f",
                request_id,
                source.pending_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE),
            )
            return
        if (error := consumer.exception()) is not None:
            logger.error("listen_ws[%s] transcription failed", request_id, exc_info=error)
        await _close_out(
            websocket,
            request_id,
            session,
            collected,
            diarize=negotiated.diarize,
            audio_sha256=digest.hexdigest(),
            frame_metadata=frame_metadata,
        )
    finally:
        for task in (consumer, reader):
            task.cancel()
        await asyncio.gather(consumer, reader, return_exceptions=True)
        source.release()


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
    timeline = await session.finalize()
    if diarize and timeline and collected:
        await send_frame(
            websocket,
            live_results(
                _attributed_words(collected, timeline),
                start=0.0,
                duration=round(session.audio_seconds, 2),
                metadata=frame_metadata,
            ),
        )
    settings = getattr(websocket.app.state, "settings", None)
    await send_frame(
        websocket,
        DeepgramLiveMetadata(
            request_id=request_id,
            sha256=audio_sha256,
            created=datetime.now(tz=UTC).isoformat(),
            duration=round(session.audio_seconds, 2),
            channels=1,
            models=[getattr(settings, "model_asr", "")] if settings else [],
            detected_language=session.detected_language,
        ),
    )
    if socket_open(websocket):
        await websocket.close(code=_CLOSE_NORMAL)
    logger.info(
        "listen_ws[%s] closed audio_s=%.2f words=%d",
        request_id,
        session.audio_seconds,
        len(collected),
    )
