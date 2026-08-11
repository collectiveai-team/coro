"""Assemble the AsyncAPI document for coro's event-driven surface.

The document is **generated, never authored**: every payload schema is derived
from the types the server serialises (see ``coro.api.asyncapi.schema``), so a
field added to a wire type appears in the published contract without anyone
editing these modules. Each channel contributes its own skeleton — address,
direction, prose — from its own module:

- ``sse_channel``: the SSE stream on POST /v1/audio/transcriptions.
- ``live_channel``: the Deepgram-native WebSocket /v1/listen.

This exists because no generator covers either one: FastAPI emits nothing into
OpenAPI for a streaming response body or for a WebSocket route, and FastStream
generates AsyncAPI from *broker* decorators, of which coro has none.
``tests/test_api_contract_documents.py`` pins both channel addresses against the
app's own routes so the two contracts cannot drift apart.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import coro

from coro.api.asyncapi import live_channel, sse_channel
from coro.api.asyncapi.models import AsyncAPIDocument, Components, Info, Server

# Re-exported for the drift guard and for coro.api.docs.
STREAM_CHANNEL_ADDRESS = sse_channel.STREAM_CHANNEL_ADDRESS
LISTEN_CHANNEL_ADDRESS = live_channel.LISTEN_CHANNEL_ADDRESS


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
    contributions = [sse_channel.contribution(), live_channel.contribution()]
    channels = {each.key: each.channel for each in contributions}
    operations = {key: value for each in contributions for key, value in each.operations.items()}
    messages = {key: value for each in contributions for key, value in each.messages.items()}
    schemas = {key: value for each in contributions for key, value in each.schemas.items()}

    return AsyncAPIDocument(
        info=Info(
            title="Coro ASR Streaming API",
            version=coro.__version__,
            description=(
                "This is the event-driven half of coro's contract: the SSE "
                "transcription stream and the Deepgram-native live socket. The "
                "request/response surface is published separately as OpenAPI at "
                "`/openapi.json`; `/docs` renders both."
            ),
        ),
        servers={"current": _server(base_url)} if base_url else {},
        channels=channels,
        operations=operations,
        components=Components(messages=messages, schemas=schemas),
    )
