"""Build the AsyncAPI document for coro's SSE transcription stream.

The document is **generated, never authored**: every payload schema is derived
from the same types the SSE writer serialises (``coro.core.models.events``
dataclasses, ``OpenAIErrorResponse``, ``SSE_TERMINATOR``), so a field added to a
wire type appears in the published contract without anyone editing this file.
Only the channel skeleton — address, direction, prose — is declared here, and
``tests/test_asyncapi_document.py`` pins the address against the OpenAPI paths
so the two contracts cannot drift apart.

This exists because no generator covers it: FastAPI emits nothing into OpenAPI
for a streaming response body, and FastStream generates AsyncAPI from *broker*
decorators, of which coro has none.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import MISSING, fields, is_dataclass
from typing import Any
from urllib.parse import urlsplit

from pydantic import TypeAdapter

import coro

from coro.api.asyncapi.models import (
    AsyncAPIDocument,
    Channel,
    Components,
    Info,
    Message,
    Operation,
    Reference,
    Server,
)
from coro.api.schemas import OpenAIErrorResponse
from coro.api.sse import SSE_MEDIA_TYPE, SSE_TERMINATOR
from coro.core.models.events import TranscriptDeltaEvent, TranscriptDoneEvent

# The channel address must equal the OpenAPI path of the route that opens the
# stream; a test asserts it against `app.openapi()` rather than trusting this.
STREAM_CHANNEL_ADDRESS = "/v1/audio/transcriptions"

_CHANNEL_KEY = "transcriptionStream"
_OPERATION_KEY = "sendTranscriptionStream"
_HTTP_BINDING_VERSION = "0.3.0"
_SCHEMA_REF_TEMPLATE = "#/components/schemas/{model}"

_CHANNEL_DESCRIPTION = (
    "Server-sent event stream returned by POST /v1/audio/transcriptions when the "
    f"`stream` form field is true. The response carries `{SSE_MEDIA_TYPE}`; each "
    "frame is a single `data:` line. Framing is OpenAI-exact: zero or more delta "
    "events, then one done event, then the terminator."
)


def _constant_wire_fields(wire_type: type) -> Iterator[tuple[str, Any]]:
    """Yield the dataclass fields the wire format always carries unchanged.

    ``init=False`` with a default means no caller can vary the value, so it is a
    constant of the message envelope (the event `type` discriminator) rather
    than a defaulted input. Reading that off the dataclass keeps the published
    discriminator in step with the code instead of restating it.
    """
    for field in fields(wire_type):
        if not field.init and field.default is not MISSING:
            yield field.name, field.default


def _payload_schema(wire_type: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive one message payload schema plus any schemas it references.

    Returns the payload schema and the definitions it ``$ref``s, which the
    caller lifts into ``components.schemas``. Both are JSON Schema documents:
    built from arbitrary wire types at runtime and handed straight to Scalar and
    redocly, so there is no fixed shape a dataclass could express.
    """
    schema = TypeAdapter(wire_type).json_schema(ref_template=_SCHEMA_REF_TEMPLATE)
    definitions = schema.pop("$defs", {})

    if is_dataclass(wire_type):
        properties = schema.get("properties", {})
        required = schema.setdefault("required", [])
        for name, value in _constant_wire_fields(wire_type):
            if name not in properties:
                continue
            # A defaulted property says "may be absent, may be anything"; the
            # wire value is neither. const + required is the accurate contract.
            properties[name] = {"const": value, "type": "string", "title": name}
            if name not in required:
                required.append(name)

    return schema, definitions


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
    delta_payload, delta_defs = _payload_schema(TranscriptDeltaEvent)
    done_payload, done_defs = _payload_schema(TranscriptDoneEvent)
    error_payload, error_defs = _payload_schema(OpenAIErrorResponse)

    messages = {
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
    return messages, schemas


def _server(base_url: str) -> Server:
    """Describe the server the stream was requested from."""
    parts = urlsplit(base_url)
    return Server(
        host=parts.netloc,
        protocol=parts.scheme or "http",
        description="The server this document was served from.",
    )


def build_asyncapi_document(base_url: str | None = None) -> AsyncAPIDocument:
    """Build the AsyncAPI document describing coro's event-driven surface.

    Args:
        base_url: Absolute URL the document is being served from. When given, it
            is published as the single server, so the document is accurate for
            the deployment that served it instead of naming an invented host.

    Returns:
        The document, ready to serialise with ``to_json()``.

    """
    messages, schemas = _messages()
    message_refs = {key: Reference(ref=f"#/components/messages/{key}") for key in messages}

    return AsyncAPIDocument(
        info=Info(
            title="Coro ASR Streaming API",
            version=coro.__version__,
            description=(
                "This is the event-driven half of coro's contract. The "
                "request/response surface is published separately as OpenAPI at "
                "`/openapi.json`; `/docs` renders both."
            ),
        ),
        servers={"current": _server(base_url)} if base_url else {},
        channels={
            _CHANNEL_KEY: Channel(
                address=STREAM_CHANNEL_ADDRESS,
                title="Transcription stream",
                description=_CHANNEL_DESCRIPTION,
                messages=message_refs,
            )
        },
        operations={
            _OPERATION_KEY: Operation(
                action="send",
                channel=Reference(ref=f"#/channels/{_CHANNEL_KEY}"),
                title="Stream transcription events",
                summary="coro sends transcript events to the requesting client.",
                messages=[
                    Reference(ref=f"#/channels/{_CHANNEL_KEY}/messages/{key}") for key in messages
                ],
                bindings={"http": {"method": "POST", "bindingVersion": _HTTP_BINDING_VERSION}},
            )
        },
        components=Components(messages=messages, schemas=schemas),
    )
