"""Packaged CLI entry point for coro.

Three subcommands, named consistently so the packaged interface is predictable:

    coro serve [flags]              start the HTTP server
    coro run INPUT [flags]          transcribe a local file, no server
    coro bench ...                  run the benchmark tooling

Server settings flags are auto-derived from ``ServerSettings`` via
pydantic-settings' ``CliSettingsSource``: every field becomes a ``--kebab-case``
flag with pydantic-settings' precedence rules — CLI flags > environment
variables > defaults. ``coro run`` accepts that same vocabulary, so there is no
second configuration language to learn.

The bare invocation is deliberately gone: ``coro`` used to mean "start the
server", which left no room for a second thing to do. See ADR 0017.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pydantic_settings import CliSettingsSource

from coro.offline import DEFAULT_RESPONSE_FORMAT
from coro.settings import ServerSettings

SUBCOMMANDS = ("serve", "run", "bench")

_USAGE = """usage: coro <command> [options]

commands:
  serve    Start the HTTP transcription server.
  run      Transcribe a local audio or video file without a server.
  bench    Run the benchmark tooling (same as coro-bench).

Run `coro <command> --help` for a command's options.
"""


def build_settings_from_cli(
    argv: list[str] | None = None, *, prog: str = "coro serve"
) -> ServerSettings:
    """Build ServerSettings honouring CLI flags, env vars, and defaults.

    Args:
        argv: Optional list of CLI arguments. Defaults to ``sys.argv[1:]``.
        prog: Program name shown in help output.

    Returns:
        A populated ``ServerSettings`` instance.

    """
    cli_source = CliSettingsSource(
        ServerSettings,
        cli_parse_args=argv if argv is not None else True,
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_prog_name=prog,
        cli_use_class_docs_for_groups=True,
    )
    return ServerSettings(_cli_settings_source=cli_source)


# MARK: Subcommand Dispatch
def split_subcommand(argv: list[str]) -> tuple[str, list[str]]:
    """Split a command name off the front of the arguments.

    Args:
        argv: Arguments after the program name.

    Returns:
        The command and the arguments that follow it.

    Raises:
        SystemExit: When no command is given, or the command is unknown. Help
            requested before any command exits successfully.

    """
    if argv and argv[0] in SUBCOMMANDS:
        return argv[0], argv[1:]

    if not argv:
        sys.stderr.write(_USAGE)
        raise SystemExit(2)
    if argv[0] in ("-h", "--help"):
        sys.stdout.write(_USAGE)
        raise SystemExit(0)

    sys.stderr.write(
        f"coro: unknown command {argv[0]!r}. Expected one of: {', '.join(SUBCOMMANDS)}.\n\n"
    )
    sys.stderr.write(_USAGE)
    raise SystemExit(2)


# MARK: serve
def serve(argv: list[str]) -> None:
    """Start the HTTP transcription server."""
    import uvicorn

    settings = build_settings_from_cli(argv)

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


# MARK: run
def _run_parser() -> argparse.ArgumentParser:
    """Build the parser for ``coro run``'s own arguments.

    Only the arguments the server has no equivalent for live here; everything
    else is a ``ServerSettings`` flag parsed from whatever this parser leaves
    behind, which is what keeps the two vocabularies identical.
    """
    parser = argparse.ArgumentParser(
        prog="coro run",
        description=(
            "Transcribe a local audio or video file. Runs in-process by default: "
            "no server, no upload, and the input file is left untouched. Accepts "
            "every `coro serve` flag in addition to those below."
        ),
        add_help=False,
    )
    parser.add_argument("input", help="Path to the audio or video file to transcribe.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the JSON response here instead of to stdout.",
    )
    parser.add_argument("--language", default=None, help="Language hint passed to the model.")
    parser.add_argument("--prompt", default=None, help="Initial prompt passed to the model.")
    parser.add_argument(
        "--response-format",
        default=DEFAULT_RESPONSE_FORMAT,
        help=(
            "Response shape, using the transcription endpoint's own values "
            f"(default: {DEFAULT_RESPONSE_FORMAT}). Rendered by the same code "
            "the endpoint uses, so output matches the server's byte for byte."
        ),
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help=(
            "Upload to an already-running server instead of transcribing "
            "in-process. Never probed for: without this flag no server is "
            "contacted. An attached run is governed by that server's "
            "configuration, so the flags above are ignored."
        ),
    )
    return parser


def run(argv: list[str]) -> None:
    """Transcribe a local file, in-process unless a server URL is given."""
    import asyncio
    from pathlib import Path

    from coro.offline import transcribe_attached, transcribe_in_process

    parser = _run_parser()
    if any(argument in ("-h", "--help") for argument in argv):
        parser.print_help()
        sys.stdout.write("\nserver settings flags:\n")
        # Exits after printing the settings-derived flags.
        build_settings_from_cli(["--help"], prog="coro run")
        return

    args, remaining = parser.parse_known_args(argv)

    source = Path(args.input)
    if not source.is_file():
        sys.stderr.write(f"coro run: no such file: {args.input}\n")
        raise SystemExit(2)

    if args.server_url is not None:
        if remaining:
            # An attached run is governed by the server's own configuration, so
            # these cannot take effect. Saying so beats letting someone believe
            # they benchmarked a backend they never actually selected.
            sys.stderr.write(
                f"coro run: ignoring {' '.join(remaining)} — an attached run uses the "
                f"server's configuration, not local flags.\n"
            )
        body, report = asyncio.run(
            transcribe_attached(
                str(source),
                server_url=args.server_url,
                language=args.language,
                prompt=args.prompt,
                response_format=args.response_format,
            )
        )
    else:
        # Unknown flags are ServerSettings flags; this rejects genuinely
        # unknown ones rather than silently ignoring them.
        settings = build_settings_from_cli(remaining, prog="coro run")
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        body, report = asyncio.run(
            transcribe_in_process(
                str(source),
                settings=settings,
                language=args.language,
                prompt=args.prompt,
                response_format=args.response_format,
            )
        )

    if args.output:
        Path(args.output).write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body + "\n")
    sys.stderr.write(f"coro run: {report.summary()}\n")


# MARK: bench
def bench(argv: list[str]) -> None:
    """Run the benchmark tooling."""
    from coro.bench.cli import main as bench_main

    bench_main(argv)


# MARK: Entry Point
_COMMANDS = {"serve": serve, "run": run, "bench": bench}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``coro`` command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command, rest = split_subcommand(arguments)
    _COMMANDS[command](rest)


if __name__ == "__main__":
    main()
