"""Published API contract: the OpenAPI and AsyncAPI documents the app serves.

These assert the *contract*, not the renderer: that both documents are served,
that the AsyncAPI document is derived from the wire types rather than restated,
and that the two documents agree on the endpoint they both describe.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from httpx import ASGITransport, AsyncClient

from coro.api.asyncapi import (
    ASYNCAPI_VERSION,
    LISTEN_CHANNEL_ADDRESS,
    STREAM_CHANNEL_ADDRESS,
    build_asyncapi_document,
)
from coro.api.deepgram.listen_ws import CLOSE_STREAM, FINALIZE, KEEP_ALIVE
from coro.api.openai.sse import SSE_TERMINATOR
from coro.app import api_metadata, create_app
from coro.core.models.events import TranscriptDeltaEvent, TranscriptDoneEvent
from coro.settings import ServerSettings


@pytest.fixture
def app():
    return create_app(ServerSettings(_env_file=None))


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _sample(event_type):
    """Build one instance of an SSE event type, whatever its single payload field."""
    if event_type is TranscriptDeltaEvent:
        return TranscriptDeltaEvent(delta="x")
    return TranscriptDoneEvent(text="x")


# MARK: Documents are served
@pytest.mark.asyncio
async def test_openapi_document_is_served(app):
    """The request/response contract is served as JSON from a stable path."""
    async with _client(app) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["openapi"].startswith("3.")


@pytest.mark.asyncio
async def test_asyncapi_document_is_served(app):
    """The event-driven contract is served as JSON from a stable path."""
    async with _client(app) as client:
        response = await client.get("/asyncapi.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["asyncapi"] == ASYNCAPI_VERSION


@pytest.mark.asyncio
async def test_asyncapi_server_describes_the_host_that_served_it(app):
    """The published server is the one the document came from, not an invented host."""
    async with _client(app) as client:
        response = await client.get("/asyncapi.json")

    servers = response.json()["servers"]
    assert [server["host"] for server in servers.values()] == ["test"]


@pytest.mark.asyncio
async def test_each_contract_points_at_the_other(app):
    """A consumer fetching one document learns the other half exists.

    The two surfaces cannot share a document, so the cross-reference is the only
    thing stopping a reader concluding that whichever document they found is the
    whole contract.
    """
    async with _client(app) as client:
        openapi = (await client.get("/openapi.json")).json()
        asyncapi = (await client.get("/asyncapi.json")).json()

    assert "/asyncapi.json" in openapi["info"]["description"]
    assert "/openapi.json" in asyncapi["info"]["description"]


# MARK: Metadata without a distribution install
def test_installed_metadata_is_published_in_the_contract():
    """With distribution metadata available, summary and license reach the document."""
    description, license_info = api_metadata("Coro does ASR.", "MIT")

    assert description.startswith("Coro does ASR.")
    assert license_info == {"name": "MIT", "identifier": "MIT"}


def test_missing_metadata_omits_the_license_rather_than_publishing_it_blank():
    """The CI contract gate exports from an uninstalled tree, so this path is live.

    An empty SPDX identifier is not a valid OpenAPI license object, and a
    description opening on a blank line is noise.
    """
    description, license_info = api_metadata("", "")

    assert license_info is None
    assert description == description.strip()
    assert "/asyncapi.json" in description


# MARK: The two contracts describe the same endpoints
@pytest.mark.asyncio
async def test_stream_channel_address_is_a_real_openapi_path(app):
    """The AsyncAPI channel address must name a path OpenAPI actually documents.

    This is the drift guard: the channel address is the one part of the AsyncAPI
    document that is declared rather than derived, so it is pinned against the
    generated OpenAPI paths instead of being trusted.
    """
    async with _client(app) as client:
        paths = (await client.get("/openapi.json")).json()["paths"]

    assert STREAM_CHANNEL_ADDRESS in paths
    assert "post" in paths[STREAM_CHANNEL_ADDRESS]


def test_live_channel_address_is_a_real_websocket_route(app):
    """The socket channel is pinned against the app's WebSocket routes.

    It cannot be pinned against OpenAPI: FastAPI emits no path for a WebSocket
    route, which is the whole reason this channel is the only published contract
    the endpoint has. The guard is therefore the route table itself.
    """
    websocket_paths = {
        route.path for route in app.routes if type(route).__name__ == "APIWebSocketRoute"
    }

    assert LISTEN_CHANNEL_ADDRESS in websocket_paths


@pytest.mark.asyncio
async def test_every_asyncapi_channel_is_pinned_to_a_route(app):
    """No channel may name an address the app does not actually serve.

    Without this, adding a channel with a typo'd or removed address publishes a
    contract for an endpoint that does not exist.
    """
    async with _client(app) as client:
        paths = set((await client.get("/openapi.json")).json()["paths"])
    paths |= {route.path for route in app.routes if type(route).__name__ == "APIWebSocketRoute"}

    addresses = {channel.address for channel in build_asyncapi_document().channels.values()}

    assert addresses <= paths


def test_live_control_vocabulary_is_the_handler_s_own(app):
    """The published control frames are the ones the handler dispatches on."""
    document = build_asyncapi_document()
    payload = document.components.messages["deepgramControl"].payload

    assert payload is not None
    assert set(payload["properties"]["type"]["enum"]) == {KEEP_ALIVE, FINALIZE, CLOSE_STREAM}


def test_live_results_payload_is_derived_from_the_frame_model(app):
    """Every field of the emitted Results frame appears in the published payload."""
    from coro.api.deepgram.live_schemas import DeepgramLiveResults

    document = build_asyncapi_document()
    payload = document.components.messages["deepgramResults"].payload

    assert payload is not None
    assert set(payload["properties"]) == set(DeepgramLiveResults.model_fields)


# MARK: Payloads are derived from the wire types
@pytest.mark.parametrize(
    ("message_key", "event_type"),
    [
        ("transcriptTextDelta", TranscriptDeltaEvent),
        ("transcriptTextDone", TranscriptDoneEvent),
    ],
)
def test_event_payload_matches_the_dataclass_that_produces_it(message_key, event_type):
    """Every field of the emitted dataclass appears in the published payload."""
    document = build_asyncapi_document()
    payload = document.components.messages[message_key].payload

    assert payload is not None
    # asdict() is exactly how coro.api.openai.sse serialises the event onto the wire,
    # so its keys are the ground truth the contract has to cover.
    assert set(payload["properties"]) == set(asdict(_sample(event_type)))


@pytest.mark.parametrize(
    ("message_key", "event_type"),
    [
        ("transcriptTextDelta", TranscriptDeltaEvent),
        ("transcriptTextDone", TranscriptDoneEvent),
    ],
)
def test_event_discriminator_is_published_as_a_required_constant(message_key, event_type):
    """`type` is `init=False`, so the contract must pin it, not default it."""
    document = build_asyncapi_document()
    payload = document.components.messages[message_key].payload

    assert payload is not None
    assert payload["properties"]["type"]["const"] == _sample(event_type).type
    assert "type" in payload["required"]


def test_terminator_payload_is_the_sentinel_the_writer_emits():
    """The `[DONE]` frame is documented from the constant the SSE writer uses."""
    document = build_asyncapi_document()
    payload = document.components.messages["streamTerminator"].payload

    assert payload is not None
    assert payload["const"] == SSE_TERMINATOR


def test_error_payload_resolves_through_components_schemas():
    """The error message `$ref`s a lifted schema rather than dangling."""
    document = build_asyncapi_document()
    payload = document.components.messages["streamError"].payload

    assert payload is not None
    ref = payload["properties"]["error"]["$ref"]
    assert ref.startswith("#/components/schemas/")
    assert ref.rsplit("/", 1)[-1] in document.components.schemas


# MARK: Serialisation
def test_document_serialises_with_asyncapi_key_spelling():
    """AsyncAPI's camelCase and `$ref` keys survive serialisation."""
    document = json.loads(build_asyncapi_document("http://example.test/").to_json())

    channel = document["channels"]["transcriptionStream"]
    assert (
        channel["messages"]["transcriptTextDelta"]["$ref"]
        == "#/components/messages/transcriptTextDelta"
    )
    delta = document["components"]["messages"]["transcriptTextDelta"]
    assert delta["contentType"] == "application/json"
    assert document["components"]["messages"]["streamTerminator"]["contentType"] == "text/plain"


def test_document_omits_servers_when_no_base_url_is_known():
    """A document built outside a request names no server rather than a fake one."""
    assert json.loads(build_asyncapi_document().to_json()).get("servers", {}) == {}
