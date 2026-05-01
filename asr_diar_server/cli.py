"""Server CLI entry point for asr-diar-server.

Configures logging and launches the packaged ASGI app via uvicorn.
Heavy model initialization happens in the lifespan, not here.

Usage:
    asr-diar-server [--host HOST] [--port PORT] [options]
"""

from __future__ import annotations

import argparse
import logging


def parse_server_args(argv=None) -> argparse.Namespace:
    """Parse asr-diar-server command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the asr-diar-server ASGI application.",
        prog="asr-diar-server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")
    parser.add_argument("--ssl-certfile", default=None, help="TLS certificate file path.")
    parser.add_argument("--ssl-keyfile", default=None, help="TLS private key file path.")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the asr-diar-server command."""
    import uvicorn

    args = parse_server_args()

    # Configure logging only from CLI/startup paths.
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn_kwargs: dict = {
        "app": "asr_diar_server.app:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": args.log_level.lower(),
        "lifespan": "on",
    }

    if args.ssl_certfile and args.ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
