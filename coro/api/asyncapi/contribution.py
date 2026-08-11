"""What one channel contributes to the assembled AsyncAPI document.

A typed boundary object rather than a tuple of dicts: each channel module hands
back exactly one of these, and the assembler merges them without knowing
anything about either channel's internals (CES-79, and the `no-dict-*`
ast-grep rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coro.api.asyncapi.models import Channel, Message, Operation


@dataclass(frozen=True)
class ChannelContribution:
    """One channel, its operations, and the components they reference.

    Attributes:
        key: The `channels` map key the channel is published under.
        channel: The channel object itself.
        operations: Operations over this channel, keyed as published.
        messages: Messages lifted into `components.messages`.
        schemas: Schemas the messages `$ref`, lifted into `components.schemas`.

    """

    key: str
    channel: Channel
    operations: dict[str, Operation] = field(default_factory=dict)
    messages: dict[str, Message] = field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
