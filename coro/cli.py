"""Server CLI entry point for coro.

CLI flags are auto-derived from ``ServerSettings`` via pydantic-settings'
``CliSettingsSource``. Every field on ``ServerSettings`` is exposed as a
``--kebab-case`` flag with the same precedence rules as pydantic-settings:

    CLI flags > environment variables > defaults

Usage:
    coro [--pipeline streaming] [--backend-diarization nemo] ...

Run ``coro --help`` to see every available flag.
"""

from __future__ import annotations

import logging

from pydantic_settings import CliSettingsSource

from coro.settings import ServerSettings


def build_settings_from_cli(argv: list[str] | None = None) -> ServerSettings:
    """Build ServerSettings honouring CLI flags, env vars, and defaults.

    Args:
        argv: Optional list of CLI arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        A populated ``ServerSettings`` instance.

    """
    cli_source = CliSettingsSource(
        ServerSettings,
        cli_parse_args=argv if argv is not None else True,
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_prog_name="coro",
        cli_use_class_docs_for_groups=True,
    )
    return ServerSettings(_cli_settings_source=cli_source)


def main() -> None:
    """Entry point for the coro command."""
    import uvicorn

    settings = build_settings_from_cli()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build the app with the parsed settings, not module-level defaults.
    from coro.app import create_app

    application = create_app(settings)

    uvicorn_kwargs: dict = {
        "app": application,
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level.lower(),
        "lifespan": "on",
    }

    if settings.ssl_certfile and settings.ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = settings.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = settings.ssl_keyfile

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
