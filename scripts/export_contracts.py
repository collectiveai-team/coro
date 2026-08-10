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

import argparse
import json
import sys
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    """Export the contracts into the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to write the documents into.")
    parser.add_argument(
        "--openapi-only",
        action="store_true",
        help="Skip the AsyncAPI document (for refs that predate it).",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = [export_openapi(args.output_dir)]
    if not args.openapi_only:
        written.append(export_asyncapi(args.output_dir))

    for path in written:
        print(f"exported {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
