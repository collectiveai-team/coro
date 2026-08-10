"""Scalar API reference, serving both published contracts from one URL.

``/docs`` renders the OpenAPI (request/response) and AsyncAPI (event stream)
documents as two sources behind a single document picker, replacing FastAPI's
Swagger UI and ReDoc — neither of which can render AsyncAPI at all.

Three ``get_scalar_api_reference()`` defaults are hostile and are overridden
here: ``scalar_js_url`` defaults to an *unversioned* jsDelivr URL that is
re-resolved on every page load with no SRI, ``telemetry`` defaults to ``True``,
and ``agent`` defaults to Scalar's hosted AI chat, which is on by default on
localhost and uploads the served document to api.scalar.com on first use. The
``api-docs-scalar-hardened`` ast-grep rule fails the build if any of the three
is left at its default.

Not overridden, and deliberately so: ``scalar_favicon_url`` is blanked (its
default fetches an image from fastapi.tiangolo.com) and ``show_developer_tools``
is left at its ``"localhost"`` default, which is local-only UI chrome that sends
nothing anywhere.

Bundle resolution is self-hosted-first: the image build drops a pinned bundle
into :data:`BUNDLE_FILENAME` and it is served from this app. A source checkout
has no bundle, so it falls back to the *version-pinned* CDN URL — still
immutable, still auditable, and it keeps ``uv run coro`` working with no build
step.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from scalar_fastapi import AgentScalarConfig, OpenAPISource, get_scalar_api_reference
from starlette.requests import Request

from coro.api.asyncapi import build_asyncapi_document

# Pinned bundle version. The image build reads SCALAR_CDN_URL out of this module
# rather than restating the version, so there is exactly one place to bump.
SCALAR_BUNDLE_VERSION = "1.64.1"
SCALAR_CDN_URL = (
    f"https://cdn.jsdelivr.net/npm/@scalar/api-reference@{SCALAR_BUNDLE_VERSION}"
    "/dist/browser/standalone.js"
)

STATIC_MOUNT_PATH = "/static"
BUNDLE_FILENAME = "scalar.standalone.js"

DOCS_PATH = "/docs"
OPENAPI_PATH = "/openapi.json"
ASYNCAPI_PATH = "/asyncapi.json"

# Scalar renders multiple `sources` as one app with a document picker; the
# sidebar only ever shows the active document, and there is no configuration
# that interleaves two documents into one navigation tree (the two specs have
# incompatible structures). The picker is therefore the *only* route to the
# event-driven contract, and by default Scalar draws it as a plain text label
# that reads like a heading rather than a control. This promotes it.
#
# The `.document-selector` class is internal to Scalar. That coupling is
# acceptable because the bundle is version-pinned, so it cannot change without a
# deliberate bump — and if a future bump does rename it, these rules go inert
# and the picker keeps working, just unstyled.
DOCUMENT_PICKER_CSS = """
.document-selector {
    padding: 12px 12px 6px !important;
}

.document-selector::before {
    content: "Contract";
    display: block;
    margin-bottom: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--scalar-sidebar-color-2);
}

.document-selector button {
    width: 100%;
    padding: 6px 10px;
    border: 1px solid var(--scalar-sidebar-border-color);
    border-radius: var(--scalar-radius);
    background: var(--scalar-background-2);
    color: var(--scalar-sidebar-color-1) !important;
}

.document-selector button:hover {
    background: var(--scalar-sidebar-item-hover-background);
    border-color: var(--scalar-color-accent);
}
"""


def static_directory() -> Path:
    """Directory the self-hosted Scalar bundle is expected in."""
    return Path(__file__).resolve().parent.parent / "static"


def bundle_path() -> Path:
    """Path of the self-hosted Scalar bundle, present or not."""
    return static_directory() / BUNDLE_FILENAME


def resolve_scalar_js_url() -> str:
    """Return the self-hosted bundle URL, or the pinned CDN URL if absent."""
    if bundle_path().is_file():
        return f"{STATIC_MOUNT_PATH}/{BUNDLE_FILENAME}"
    return SCALAR_CDN_URL


def register_docs(app: FastAPI, *, title: str) -> None:
    """Register the AsyncAPI document and the Scalar reference on ``app``.

    Args:
        app: Application to register on. It must have been constructed with
            ``docs_url=None`` and ``redoc_url=None``; Scalar takes ``/docs``.
        title: Browser tab title for the reference page.

    """
    directory = static_directory()
    if directory.is_dir():
        app.mount(STATIC_MOUNT_PATH, StaticFiles(directory=directory), name="static")

    @app.get(ASYNCAPI_PATH, include_in_schema=False)
    async def asyncapi_document(request: Request) -> Response:
        """Serve the generated AsyncAPI document for the SSE surface."""
        document = build_asyncapi_document(str(request.base_url))
        return Response(content=document.to_json(), media_type="application/json")

    @app.get(DOCS_PATH, include_in_schema=False)
    async def scalar_reference() -> HTMLResponse:
        """Render both contracts in one Scalar reference."""
        return get_scalar_api_reference(
            sources=[
                OpenAPISource(title="REST", slug="rest", url=OPENAPI_PATH, default=True),
                OpenAPISource(title="Events", slug="events", url=ASYNCAPI_PATH),
            ],
            title=title,
            custom_css=DOCUMENT_PICKER_CSS,
            scalar_js_url=resolve_scalar_js_url(),
            # Agent Scalar is an AI chat panel wired to Scalar's hosted service
            # (api.scalar.com via proxy.scalar.com). It is enabled by default on
            # localhost and, per Scalar's own docs, "your OpenAPI document is
            # uploaded on first message" — so leaving the default in place lets
            # anyone who opens /docs hand this server's API surface to a third
            # party. Off everywhere, not just in production.
            agent=AgentScalarConfig(disabled=True),
            # Defaults to True, which pulls Inter and JetBrains Mono from
            # fonts.scalar.com on every page load. Self-hosting the bundle is
            # pointless if the page still calls out for webfonts, and an
            # air-gapped deployment would just block on them.
            with_default_fonts=False,
            # Default is fastapi.tiangolo.com — a third-party fetch on every
            # page load for a favicon. Empty is served from nothing at all.
            scalar_favicon_url="",
            telemetry=False,
        )
