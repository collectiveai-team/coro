"""Generated AsyncAPI contract for coro's event-driven surface.

Two channels: the SSE transcription stream and the Deepgram-native live socket.
"""

from coro.api.asyncapi.document import (
    LISTEN_CHANNEL_ADDRESS,
    STREAM_CHANNEL_ADDRESS,
    build_asyncapi_document,
)
from coro.api.asyncapi.models import ASYNCAPI_VERSION, AsyncAPIDocument

__all__ = [
    "ASYNCAPI_VERSION",
    "LISTEN_CHANNEL_ADDRESS",
    "STREAM_CHANNEL_ADDRESS",
    "AsyncAPIDocument",
    "build_asyncapi_document",
]
