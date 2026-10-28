"""Export coro's generated API contracts as files for CI to lint and diff.

Both documents are produced by the running app — OpenAPI by FastAPI from the
route decorators, AsyncAPI by ``coro.api.asyncapi`` from the SSE wire types — so
this only writes them out. The results are build artifacts: gitignored, linted
and diffed in CI, never committed.

Usage::

    python scripts/export_contracts.py <output-dir> [--openapi-only]

``--openapi-only`` exists for the breaking-change gate, which re-runs this
against a base ref that predates the AsyncAPI document.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from coro.cli_support import invoke, new_app, out_console


def export_openapi(output_dir: Path) -> Path:
    """Write the OpenAPI document FastAPI generates from the routes."""
    from coro.app import app

    target = output_dir / "openapi.json"
    target.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    return target


def export_asyncapi(output_dir: Path) -> Path:
    """Write the AsyncAPI document generated from the SSE wire types."""
    from coro.api.asyncapi import build_asyncapi_document

    target = output_dir / "asyncapi.json"
    target.write_text(build_asyncapi_document().to_json() + "\n", encoding="utf-8")
    return target


app = new_app("export_contracts", help=__doc__)


@app.default
def _export(output_dir: Path, /, *, openapi_only: bool = False) -> None:
    """Export the contracts into the requested directory.

    Args:
        output_dir: Directory to write the documents into.
        openapi_only: Skip the AsyncAPI document (for refs that predate it).

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [export_openapi(output_dir)]
    if not openapi_only:
        written.append(export_asyncapi(output_dir))

    for path in written:
        out_console.out(f"exported {path} ({path.stat().st_size} bytes)")


def main(argv: Sequence[str] | None = None) -> int:
    """Export the contracts into the requested directory."""
    invoke(app, argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
