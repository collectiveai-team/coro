"""Wire-type to JSON Schema derivation shared by the AsyncAPI channels.

Every published payload is derived from the type the server actually
serialises, so a field added to a wire type appears in the contract without
anyone editing a channel module. This is the one place that derivation lives.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import MISSING, fields, is_dataclass
from typing import Any

from pydantic import TypeAdapter

SCHEMA_REF_TEMPLATE = "#/components/schemas/{model}"


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


def payload_schema(wire_type: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive one message payload schema plus any schemas it references.

    Returns the payload schema and the definitions it ``$ref``s, which the
    caller lifts into ``components.schemas``. Both are JSON Schema documents:
    built from arbitrary wire types at runtime and handed straight to Scalar and
    redocly, so there is no fixed shape a dataclass could express.
    """
    schema = TypeAdapter(wire_type).json_schema(ref_template=SCHEMA_REF_TEMPLATE)
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
