"""The Deepgram-native live socket, as one AsyncAPI channel.

`WebSocket /v1/listen` is the only part of coro's contract that FastAPI's
OpenAPI generator cannot describe at all: OpenAPI has no notion of a socket, so
without this the endpoint would ship with a test suite and no published
contract. AsyncAPI is where a bidirectional stream belongs (ADR 0013, ADR 0015).

Both directions are published, because both are part of the vendor contract the
endpoint claims to implement: coro *sends* Results/Metadata/Error frames, and
*receives* binary audio plus in-band JSON control frames. Everything is derived
from code — payloads from the ``live_schemas`` models, the control vocabulary
from the constants the handler compares against, the format bounds from
``coro.pcm`` — so the document cannot drift from the handler.
"""

from __future__ import annotations

from typing import Any

from coro.api.asyncapi.contribution import ChannelContribution
from coro.api.asyncapi.models import Channel, Message, Operation, Reference
from coro.api.asyncapi.schema import payload_schema
from coro.api.deepgram.listen_ws import CLOSE_STREAM, FINALIZE, KEEP_ALIVE
from coro.api.deepgram.live_schemas import (
    DeepgramLiveError,
    DeepgramLiveMetadata,
    DeepgramLiveResults,
)
from coro.pcm import LINEAR16, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE

# Must equal the path of the WebSocket route. It is deliberately *not* an
# OpenAPI path — no such path exists — so the drift guard pins it against the
# app's WebSocket routes instead.
LISTEN_CHANNEL_ADDRESS = "/v1/listen"

CHANNEL_KEY = "deepgramLiveStream"
SEND_OPERATION_KEY = "sendDeepgramLiveFrames"
RECEIVE_OPERATION_KEY = "receiveDeepgramLiveAudio"

_WS_BINDING_VERSION = "0.1.0"

# Frames coro puts on the socket, and frames it takes off it. Split because the
# two directions become two AsyncAPI operations over the one channel.
SERVER_MESSAGE_KEYS = ["deepgramResults", "deepgramMetadata", "deepgramError"]
CLIENT_MESSAGE_KEYS = ["deepgramAudioFrame", "deepgramControl"]

_CHANNEL_DESCRIPTION = (
    "Deepgram-native live transcription socket. The client declares its audio "
    f"format in the query string (`encoding={LINEAR16}`, `sample_rate` between "
    f"{MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, `channels=1`, plus `diarize` "
    "and `language`), streams raw samples as binary frames, and receives one "
    "`Results` frame per completed ASR window followed by a closing `Metadata` "
    "frame. A rejected declaration produces one `Error` frame and a 1008 close, "
    "before any audio is accepted."
)

# Raw PCM has no JSON Schema type of its own; `string` + a binary content
# encoding is how JSON Schema spells an opaque byte payload.
_AUDIO_FRAME_SCHEMA: dict[str, Any] = {
    "type": "string",
    "contentEncoding": "binary",
    "title": "PcmAudioFrame",
    "description": (
        "Raw little-endian 16-bit mono PCM at the declared `sample_rate`. Any "
        "frame size is accepted; coro resamples to its internal 16 kHz rate."
    ),
}

# Control vocabulary read off the constants the handler dispatches on, so a new
# control frame cannot be added to the handler without appearing here.
_CONTROL_FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "LiveControlFrame",
    "properties": {"type": {"type": "string", "enum": [KEEP_ALIVE, FINALIZE, CLOSE_STREAM]}},
    "required": ["type"],
    "description": (
        f"`{KEEP_ALIVE}` holds the socket open without ending the stream. "
        f"`{FINALIZE}` and `{CLOSE_STREAM}` both end it, flushing the audio "
        "buffered so far before the closing `Metadata` frame."
    ),
}


def _messages() -> tuple[dict[str, Message], dict[str, dict[str, Any]]]:
    """Build every message on the live channel, plus the schemas they reference."""
    results_payload, results_defs = payload_schema(DeepgramLiveResults)
    metadata_payload, metadata_defs = payload_schema(DeepgramLiveMetadata)
    error_payload, error_defs = payload_schema(DeepgramLiveError)

    built = {
        "deepgramResults": Message(
            name="deepgramResults",
            title="Live results",
            summary=(
                "One completed ASR window. Always `is_final`: coro surfaces only "
                "tokens it has already accepted, so there are no interim frames "
                "to revise. Under `diarize=true` a final frame replays every word "
                "with its speaker attached, because a streaming diarizer has no "
                "timeline until the audio ends."
            ),
            content_type="application/json",
            payload=results_payload,
        ),
        "deepgramMetadata": Message(
            name="deepgramMetadata",
            title="Live metadata",
            summary=(
                "Closes every successful stream. `sha256` digests the audio "
                "actually received, accumulated as it streamed."
            ),
            content_type="application/json",
            payload=metadata_payload,
        ),
        "deepgramError": Message(
            name="deepgramError",
            title="Live error",
            summary=(
                "Sent instead of results when the declared format cannot be "
                "ingested, or when transcription fails mid-stream. The socket "
                "closes after it."
            ),
            content_type="application/json",
            payload=error_payload,
        ),
        "deepgramAudioFrame": Message(
            name="deepgramAudioFrame",
            title="Audio frame",
            summary="One binary frame of the client's PCM stream.",
            content_type="application/octet-stream",
            payload=_AUDIO_FRAME_SCHEMA,
        ),
        "deepgramControl": Message(
            name="deepgramControl",
            title="Control frame",
            summary="In-band JSON control, interleaved with the audio frames.",
            content_type="application/json",
            payload=_CONTROL_FRAME_SCHEMA,
        ),
    }

    schemas = {**results_defs, **metadata_defs, **error_defs}
    return built, schemas


# The handshake, described where the WebSocket binding spec puts it: on the
# channel. Operation-level `ws` bindings accept no fields at all, so the
# parameters the handler reads off the query string can only be published here.
_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "LiveQueryParameters",
    "properties": {
        "encoding": {
            "type": "string",
            "enum": [LINEAR16],
            "description": f"Only `{LINEAR16}` is accepted. Absent means `{LINEAR16}`.",
        },
        "sample_rate": {
            "type": "integer",
            "minimum": MIN_SAMPLE_RATE,
            "maximum": MAX_SAMPLE_RATE,
            "description": "Rate of the submitted PCM. Resampled to coro's internal rate.",
        },
        "channels": {"type": "integer", "enum": [1], "description": "Mono only."},
        "diarize": {
            "type": "boolean",
            "description": "Attach per-word speakers to the final Results frame.",
        },
        "language": {"type": "string", "description": "Passed through to the ASR backend."},
    },
}

_WS_CHANNEL_BINDING: dict[str, Any] = {
    "ws": {"method": "GET", "query": _QUERY_SCHEMA, "bindingVersion": _WS_BINDING_VERSION}
}


def _channel() -> Channel:
    """Describe the socket both directions travel over."""
    keys = [*SERVER_MESSAGE_KEYS, *CLIENT_MESSAGE_KEYS]
    return Channel(
        address=LISTEN_CHANNEL_ADDRESS,
        title="Deepgram live stream",
        description=_CHANNEL_DESCRIPTION,
        messages={key: Reference(ref=f"#/components/messages/{key}") for key in keys},
        bindings=_WS_CHANNEL_BINDING,
    )


def _refs(keys: list[str]) -> list[Reference]:
    return [Reference(ref=f"#/channels/{CHANNEL_KEY}/messages/{key}") for key in keys]


def _operations() -> tuple[tuple[str, Operation], ...]:
    """Describe both directions: coro sends frames and receives audio.

    Neither carries a `ws` binding: the binding spec defines fields for channels
    only, and redocly rejects an operation-level one outright.
    """
    return (
        (
            SEND_OPERATION_KEY,
            Operation(
                action="send",
                channel=Reference(ref=f"#/channels/{CHANNEL_KEY}"),
                title="Send live transcription frames",
                summary="coro sends results, then metadata, to the connected client.",
                messages=_refs(SERVER_MESSAGE_KEYS),
            ),
        ),
        (
            RECEIVE_OPERATION_KEY,
            Operation(
                action="receive",
                channel=Reference(ref=f"#/channels/{CHANNEL_KEY}"),
                title="Receive live audio and control",
                summary="coro receives the client's PCM frames and control frames.",
                messages=_refs(CLIENT_MESSAGE_KEYS),
            ),
        ),
    )


def contribution() -> ChannelContribution:
    """Assemble this channel's whole contribution to the document."""
    messages, schemas = _messages()
    operations = dict(_operations())
    return ChannelContribution(
        key=CHANNEL_KEY,
        channel=_channel(),
        operations=operations,
        messages=messages,
        schemas=schemas,
    )
