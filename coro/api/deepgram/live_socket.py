"""The inbound side of WebSocket /v1/listen: reading frames, control, admission.

Kept apart from the route so the handler only orchestrates: this module owns
how the socket is read (never waiting on the pipeline, ADR 0023), Deepgram's
in-band control frames, and the per-client rate limits (ADR 0024).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math

from fastapi import WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from coro.api.deepgram.live_schemas import DeepgramLiveError
from coro.api.rate_limit import (
    AUDIO_EXCEEDED_MESSAGE,
    admit_request,
    audio_quota_empty,
    client_key,
    rate_limits_of,
)
from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from coro.pcm import PcmStreamConverter
from coro.pipelines.live import LiveAudioSource
from coro.pipelines.spill import SpillDirectoryError, resolve_spill_dir

logger = logging.getLogger(__name__)

CLOSE_STREAM = "CloseStream"
FINALIZE = "Finalize"
KEEP_ALIVE = "KeepAlive"

# 1008 Policy Violation: a rejected declaration, or a client over its limits.
CLOSE_POLICY = 1008

RATE_LIMITED_DESCRIPTION = "Rate limit exceeded"


def socket_open(websocket: WebSocket) -> bool:
    """Return True while neither the client nor this server has closed the socket."""
    return (
        websocket.client_state is WebSocketState.CONNECTED
        and websocket.application_state is WebSocketState.CONNECTED
    )


async def send_frame(websocket: WebSocket, model) -> None:
    """Send a JSON frame, or nothing once either side has closed."""
    if socket_open(websocket):
        await websocket.send_text(model.model_dump_json(exclude_none=True))


async def reject(websocket: WebSocket, *, description: str, message: str) -> None:
    """Send an ``Error`` frame and close with a policy-violation code."""
    await send_frame(websocket, DeepgramLiveError(description=description, message=message))
    await websocket.close(code=CLOSE_POLICY)


async def deny_if_rate_limited(websocket: WebSocket) -> bool:
    """Refuse the upgrade with HTTP 429 when the client is over its limits.

    Checked before ``accept``, so a limited client gets a plain HTTP response
    with ``Retry-After`` rather than an open socket. Returns True if denied.
    """
    limited = admit_request(websocket) or audio_quota_empty(websocket)
    if limited is None:
        return False
    body = {"err_code": "TOO_MANY_REQUESTS", "err_msg": limited.message}
    headers = {"Retry-After": str(max(1, math.ceil(limited.retry_after_seconds)))}
    await websocket.send_denial_response(JSONResponse(body, status_code=429, headers=headers))
    return True


def _over_audio_quota(websocket: WebSocket, pcm: bytes) -> bool:
    """Charge ``pcm`` to the client's audio quota; True if it does not fit."""
    limits = rate_limits_of(websocket)
    if limits is None:
        return False
    seconds = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    return limits.charge_audio(client_key(websocket), seconds) is not None


async def read_socket(
    websocket: WebSocket,
    source: LiveAudioSource,
    converter: PcmStreamConverter,
    digest: hashlib._Hash,
    request_id: str,
) -> None:
    """Read every inbound frame until the socket disconnects.

    Reading never waits on the pipeline: uvicorn stops reading the TCP stream
    while a message is unconsumed, and keepalive pongs share that stream, so a
    handler that stops reading while it is behind gets the socket closed with
    ``1011 keepalive ping timeout`` (ADR 0023). ``CloseStream`` ends the audio
    but reading continues, so pings are still answered while the backlog
    drains; frames after it are ignored.

    Audio is charged against the client's audio quota as it arrives; a stream
    that runs out is closed with an ``Error`` frame and code 1008.

    The digest accumulates as audio arrives: a live stream has no complete
    payload to hash up front, but ``Metadata`` must still report one.
    """
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if source.closed:
                continue
            if (payload := message.get("bytes")) is not None:
                digest.update(payload)
                pcm = converter.push(payload)
                if _over_audio_quota(websocket, pcm):
                    logger.info("listen_ws[%s] audio quota exhausted; closing", request_id)
                    # Close the socket before ending the audio, so the handler
                    # sees an abandoned stream rather than one to close out.
                    await reject(
                        websocket,
                        description=RATE_LIMITED_DESCRIPTION,
                        message=AUDIO_EXCEEDED_MESSAGE,
                    )
                    await source.close()
                    continue
                await source.push(pcm)
                continue
            text = message.get("text")
            if text is not None and control_type(text) in {CLOSE_STREAM, FINALIZE}:
                await source.push(converter.flush())
                await source.close()
            # KeepAlive and unrecognised control frames hold the socket open.
    finally:
        await source.close()


def control_type(text: str) -> str:
    """Return the ``type`` of a JSON control frame, or '' if unparseable."""
    try:
        return str(json.loads(text).get("type", ""))
    except (ValueError, AttributeError):
        return ""


def spill_dir(websocket: WebSocket) -> str | None:
    """Real-disk directory for the live audio backlog.

    The Streaming Pipeline's transcript spill dir is already resolved at
    startup; other pipelines resolve the same default, and fall back to the
    system temp dir when no real-disk candidate exists.
    """
    settings = getattr(websocket.app.state, "settings", None)
    configured = getattr(settings, "transcript_spill_dir", None)
    if configured is not None:
        return configured
    try:
        return resolve_spill_dir(None)
    except SpillDirectoryError:
        return None
