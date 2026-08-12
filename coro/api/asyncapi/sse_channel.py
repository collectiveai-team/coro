"""The SSE transcription stream, as one AsyncAPI channel.

Payloads are derived from the same types the SSE writer serialises
(``coro.core.models.events`` dataclasses, ``OpenAIErrorResponse``,
``SSE_TERMINATOR``), so this module declares only the channel skeleton —
address, direction, prose.
"""

from __future__ import annotations

from typing import Any

from coro.api.asyncapi.contribution import ChannelContribution
from coro.api.asyncapi.models import Channel, Message, Operation, Reference
from coro.api.asyncapi.schema import payload_schema
from coro.api.openai.schemas import OpenAIErrorResponse
from coro.api.openai.sse import SSE_MEDIA_TYPE, SSE_TERMINATOR
from coro.core.models.events import TranscriptDeltaEvent, TranscriptDoneEvent

# The channel address must equal the OpenAPI path of the route that opens the
# stream; a test asserts it against `app.openapi()` rather than trusting this.
STREAM_CHANNEL_ADDRESS = "/v1/audio/transcriptions"

CHANNEL_KEY = "transcriptionStream"
OPERATION_KEY = "sendTranscriptionStream"

_HTTP_BINDING_VERSION = "0.3.0"

_CHANNEL_DESCRIPTION = (
    "Server-sent event stream returned by POST /v1/audio/transcriptions when the "
    f"`stream` form field is true. The response carries `{SSE_MEDIA_TYPE}`; each "
    "frame is a single `data:` line. Framing is OpenAI-exact: zero or more delta "
    "events, then one done event, then the terminator."
)

# The terminator has no wire dataclass to derive from — it is a bare string —
# so its schema is stated once here, pinned to the constant the SSE writer uses.
_TERMINATOR_SCHEMA: dict[str, Any] = {
    "type": "string",
    "const": SSE_TERMINATOR,
    "title": "StreamTerminator",
    "description": "Literal sentinel, not JSON.",
}


def _messages() -> tuple[dict[str, Message], dict[str, dict[str, Any]]]:
    """Build every message on the stream channel, plus the schemas they reference."""
    delta_payload, delta_defs = payload_schema(TranscriptDeltaEvent)
    done_payload, done_defs = payload_schema(TranscriptDoneEvent)
    error_payload, error_defs = payload_schema(OpenAIErrorResponse)

    built = {
        "transcriptTextDelta": Message(
            name="transcriptTextDelta",
            title="Transcript text delta",
            summary="Incremental transcript text. Emitted zero or more times, in order.",
            content_type="application/json",
            payload=delta_payload,
        ),
        "transcriptTextDone": Message(
            name="transcriptTextDone",
            title="Transcript text done",
            summary=(
                "Final transcript. `text` carries the complete transcription "
                "response as a JSON-encoded string, so consumers must parse it a "
                "second time."
            ),
            content_type="application/json",
            payload=done_payload,
        ),
        "streamError": Message(
            name="streamError",
            title="Stream error",
            summary=(
                "Replaces the remaining events when the stream fails. Response "
                "headers are already on the wire by then, so a mid-stream failure "
                "surfaces here and never as an HTTP status code."
            ),
            content_type="application/json",
            payload=error_payload,
        ),
        "streamTerminator": Message(
            name="streamTerminator",
            title="Stream terminator",
            summary="Closes every stream, successful or failed.",
            content_type="text/plain",
            payload=_TERMINATOR_SCHEMA,
        ),
    }

    schemas = {**delta_defs, **done_defs, **error_defs}
    return built, schemas


def _channel(message_keys: list[str]) -> Channel:
    """Describe the channel the SSE events travel over."""
    return Channel(
        address=STREAM_CHANNEL_ADDRESS,
        title="Transcription stream",
        description=_CHANNEL_DESCRIPTION,
        messages={key: Reference(ref=f"#/components/messages/{key}") for key in message_keys},
    )


def _operations(message_keys: list[str]) -> tuple[tuple[str, Operation], ...]:
    """Describe what coro does with the channel: it sends every event on it.

    Returned as key/operation pairs; ``contribution`` builds the published map.
    """
    return (
        (
            OPERATION_KEY,
            Operation(
                action="send",
                channel=Reference(ref=f"#/channels/{CHANNEL_KEY}"),
                title="Stream transcription events",
                summary="coro sends transcript events to the requesting client.",
                messages=[
                    Reference(ref=f"#/channels/{CHANNEL_KEY}/messages/{key}")
                    for key in message_keys
                ],
                bindings={"http": {"method": "POST", "bindingVersion": _HTTP_BINDING_VERSION}},
            ),
        ),
    )


def contribution() -> ChannelContribution:
    """Assemble this channel's whole contribution to the document."""
    messages, schemas = _messages()
    keys = list(messages)
    operations = dict(_operations(keys))
    return ChannelContribution(
        key=CHANNEL_KEY,
        channel=_channel(keys),
        operations=operations,
        messages=messages,
        schemas=schemas,
    )
