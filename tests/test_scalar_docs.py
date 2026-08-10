"""Scalar API reference behavior: both contracts, hardened defaults, no Swagger."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from coro.api import docs as docs_module
from coro.api.docs import DOCUMENT_PICKER_CSS, SCALAR_BUNDLE_VERSION, SCALAR_CDN_URL
from coro.app import create_app
from coro.settings import ServerSettings


@pytest.fixture
def app():
    return create_app(ServerSettings(_env_file=None))


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_docs_page_offers_both_contracts(app):
    """The reference is configured with the REST and event documents as sources."""
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    assert "/openapi.json" in html
    assert "/asyncapi.json" in html


@pytest.mark.asyncio
async def test_docs_page_disables_telemetry(app):
    """`telemetry` defaults to True upstream; the rendered config must say false."""
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    assert '"telemetry": false' in html


@pytest.mark.asyncio
async def test_docs_page_disables_the_hosted_ai_agent(app):
    """Agent Scalar is on by default on localhost and uploads the document.

    Per Scalar's configuration docs the served OpenAPI document is uploaded to
    api.scalar.com on the first message, so anyone who opens /docs could hand
    this server's API surface to a third party.
    """
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    assert '"agent": {"disabled": true}' in html
    assert "Ask AI" not in html


@pytest.mark.asyncio
async def test_docs_page_loads_no_third_party_webfonts(app):
    """Scalar pulls Inter/JetBrains Mono from fonts.scalar.com unless told not to.

    Self-hosting the bundle buys nothing if the page still calls out for fonts,
    and an air-gapped deployment would block on them.
    """
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    assert '"withDefaultFonts": false' in html
    assert "fonts.scalar.com" not in html


@pytest.mark.asyncio
async def test_docs_page_references_no_scalar_hosted_service(app, monkeypatch):
    """With the bundle self-hosted, the page must name no scalar.com origin at all."""
    monkeypatch.setattr(docs_module, "resolve_scalar_js_url", lambda: "/static/bundle.js")
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    assert "scalar.com" not in html


@pytest.mark.asyncio
async def test_docs_page_never_loads_an_unversioned_bundle(app):
    """The upstream default re-resolves an unpinned CDN URL on every page load."""
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    assert 'cdn.jsdelivr.net/npm/@scalar/api-reference"' not in html
    assert "cdn.jsdelivr.net/npm/@scalar/api-reference/" not in html


@pytest.mark.asyncio
async def test_swagger_and_redoc_are_gone(app):
    """Scalar owns /docs, so FastAPI's own renderers must not also be mounted."""
    async with _client(app) as client:
        redoc = await client.get("/redoc")
        swagger_oauth = await client.get("/docs/oauth2-redirect")

    assert redoc.status_code == 404
    assert swagger_oauth.status_code == 404


@pytest.mark.asyncio
async def test_document_picker_is_styled_as_a_control(app):
    """The picker is the only route to the event contract, so it must not read as a heading.

    Scalar renders multiple sources behind a picker and shows only the active
    document's sidebar; without this styling the switch is plain text that is
    easy to miss entirely.
    """
    async with _client(app) as client:
        html = (await client.get("/docs")).text

    # Scalar embeds custom CSS as a JSON string inside the config literal, so
    # the stylesheet reaches the page escaped rather than verbatim.
    assert json.dumps(DOCUMENT_PICKER_CSS)[1:-1] in html


def test_cdn_fallback_url_is_version_pinned():
    """The fallback is immutable: a pinned release, not a floating tag."""
    assert f"@scalar/api-reference@{SCALAR_BUNDLE_VERSION}" in SCALAR_CDN_URL


def test_self_hosted_bundle_is_preferred_when_present(tmp_path, monkeypatch):
    """A bundle dropped by the image build is served locally, not from a CDN."""
    bundle = tmp_path / "scalar.standalone.js"
    bundle.write_text("/* bundle */")
    monkeypatch.setattr(docs_module, "bundle_path", lambda: bundle)

    assert docs_module.resolve_scalar_js_url() == "/static/scalar.standalone.js"


def test_cdn_is_used_when_no_bundle_was_built(tmp_path, monkeypatch):
    """A source checkout has no bundle, so /docs still works off the pinned CDN."""
    monkeypatch.setattr(docs_module, "bundle_path", lambda: tmp_path / "absent.js")

    assert docs_module.resolve_scalar_js_url() == SCALAR_CDN_URL
