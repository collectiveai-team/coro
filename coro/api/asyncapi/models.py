"""Typed models for the AsyncAPI 3.0 subset coro emits.

Only the object types the generated document actually uses are modelled. A
partial, strict model is deliberate: it makes an unrepresentable document a
construction-time error rather than a lint finding three CI steps later, and it
keeps the emitted document a *typed boundary object* instead of a nested raw
dict (CES-79).

AsyncAPI spells its keys in camelCase and uses ``$ref``; both are handled with
field aliases so the Python side stays snake_case. Serialisation therefore MUST
go through :meth:`AsyncAPIDocument.to_json`, which applies ``by_alias`` and
drops unset optionals.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The spec version the emitted document declares. FastStream emits 3.0.0/2.6.0
# only, so 3.0.0 is also the interoperable choice if a broker surface is ever
# added alongside this one.
ASYNCAPI_VERSION = "3.0.0"


class _Node(BaseModel):
    """Base for every emitted AsyncAPI object."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Reference(_Node):
    """A ``$ref`` pointer into ``components``."""

    ref: Annotated[str, Field(alias="$ref")]


class Info(_Node):
    """Document-level metadata (``info``)."""

    title: str
    version: str
    description: str | None = None


class Server(_Node):
    """A connection endpoint the channels are reachable through."""

    host: str
    protocol: str
    description: str | None = None


class Message(_Node):
    """One message shape that can travel over a channel."""

    name: str
    title: str
    summary: str | None = None
    content_type: Annotated[str | None, Field(alias="contentType")] = None
    # JSON Schema is genuinely a free-form document here: it is derived from the
    # wire types at build time, so there is no fixed shape to model.
    payload: dict[str, Any] | None = None
    bindings: dict[str, Any] | None = None


class Channel(_Node):
    """An addressable stream messages are exchanged over.

    ``bindings`` is where a protocol describes the connection itself — the
    WebSocket binding's handshake method and query schema live here, not on the
    operations, which the binding spec leaves empty.
    """

    address: str
    title: str | None = None
    description: str | None = None
    messages: dict[str, Reference] = Field(default_factory=dict)
    servers: list[Reference] | None = None
    bindings: dict[str, Any] | None = None


class Operation(_Node):
    """What the application does with a channel.

    ``send`` means *this* application puts the messages on the channel, which is
    the direction of an SSE response stream.
    """

    action: Literal["send", "receive"]
    channel: Reference
    title: str | None = None
    summary: str | None = None
    description: str | None = None
    messages: list[Reference] = Field(default_factory=list)
    bindings: dict[str, Any] | None = None


class Components(_Node):
    """Reusable message and schema definitions."""

    messages: dict[str, Message] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)


class AsyncAPIDocument(_Node):
    """A complete AsyncAPI 3.0 document."""

    asyncapi: str = ASYNCAPI_VERSION
    info: Info
    servers: dict[str, Server] = Field(default_factory=dict)
    channels: dict[str, Channel] = Field(default_factory=dict)
    operations: dict[str, Operation] = Field(default_factory=dict)
    components: Components = Field(default_factory=Components)

    def to_json(self) -> str:
        """Render the document as the JSON served at ``/asyncapi.json``.

        Returns the serialised string rather than a mapping so the one correct
        set of serialisation options (alias keys, no unset optionals) cannot be
        forgotten by a caller, and so the CI exporter and the HTTP route emit
        byte-identical documents.
        """
        return self.model_dump_json(by_alias=True, exclude_none=True, indent=2)
