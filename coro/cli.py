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

import sys
from collections.abc import Sequence
from typing import Annotated, NoReturn

from cyclopts import Parameter

from coro.cli_support import invoke, new_app
from coro.core.logger import configure_logging
from coro.offline import DEFAULT_RESPONSE_FORMAT
from coro.settings.cli import build_settings_from_cli

SUBCOMMANDS = ("serve", "run", "bench")

_USAGE = """usage: coro <command> [options]

commands:
  serve    Start the HTTP transcription server.
  run      Transcribe a local audio or video file without a server.
  bench    Run the benchmark tooling (same as coro-bench).

Run `coro <command> --help` for a command's options.
"""

# Everything after the command name belongs to that command's own parser.
Forwarded = Annotated[str, Parameter(allow_leading_hyphen=True)]


# MARK: serve
def serve(argv: list[str]) -> None:
    """Start the HTTP transcription server."""
    import uvicorn

    settings = build_settings_from_cli(argv)

    configure_logging(settings.log_level)

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
_run_app = new_app(
    "coro run",
    help=(
        "Transcribe a local audio or video file. Runs in-process by default: "
        "no server, no upload, and the input file is left untouched. Accepts "
        "every `coro serve` flag in addition to those below."
    ),
    help_flags=(),
)


@_run_app.default
def _transcribe(
    input: str,  # noqa: A002
    /,
    *settings_flags: Forwarded,
    output: Annotated[str | None, Parameter(name=["--output", "-o"])] = None,
    language: str | None = None,
    prompt: str | None = None,
    response_format: Annotated[
        str,
        Parameter(
            help="Response shape, using the transcription endpoint's own values. Rendered "
            "by the same code the endpoint uses, so output matches the server's byte for byte."
        ),
    ] = DEFAULT_RESPONSE_FORMAT,
    server_url: Annotated[
        str | None,
        Parameter(
            help="Upload to an already-running server instead of transcribing "
            "in-process. Never probed for: without this flag no server is "
            "contacted. An attached run is governed by that server's "
            "configuration, so the flags above are ignored."
        ),
    ] = None,
) -> None:
    """Transcribe a local file, in-process unless a server URL is given.

    Only the arguments the server has no equivalent for are parameters here;
    everything else is a ``ServerSettings`` flag collected in ``settings_flags``,
    which is what keeps the two vocabularies identical.

    Args:
        input: Path to the audio or video file to transcribe.
        settings_flags: coro serve flags for an in-process run.
        output: Write the JSON response here instead of to stdout.
        language: Language hint passed to the model.
        prompt: Initial prompt passed to the model.
        response_format: Response shape.
        server_url: Server to upload to instead of transcribing in-process.

    """
    import asyncio
    from pathlib import Path

    from coro.offline import transcribe_attached, transcribe_in_process

    source = Path(input)
    if not source.is_file():
        sys.stderr.write(f"coro run: no such file: {input}\n")
        raise SystemExit(2)

    remaining = list(settings_flags)
    if server_url is not None:
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
                server_url=server_url,
                language=language,
                prompt=prompt,
                response_format=response_format,
            )
        )
    else:
        # Unknown flags are ServerSettings flags; this rejects genuinely
        # unknown ones rather than silently ignoring them.
        settings = build_settings_from_cli(remaining, prog="coro run")
        configure_logging(settings.log_level)
        body, report = asyncio.run(
            transcribe_in_process(
                str(source),
                settings=settings,
                language=language,
                prompt=prompt,
                response_format=response_format,
            )
        )

    if output:
        Path(output).write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body + "\n")
    sys.stderr.write(f"coro run: {report.summary()}\n")


def run(argv: list[str]) -> None:
    """Transcribe a local file, in-process unless a server URL is given."""
    if any(argument in ("-h", "--help") for argument in argv):
        _run_app.help_print([])
        sys.stdout.write("\nserver settings flags:\n")
        # Exits after printing the settings-derived flags.
        build_settings_from_cli(["--help"], prog="coro run")
        return
    invoke(_run_app, argv)


# MARK: bench
def bench(argv: list[str]) -> None:
    """Run the benchmark tooling."""
    from coro.bench.cli import main as bench_main

    bench_main(argv)


# MARK: Entry Point
app = new_app(
    "coro",
    usage="Usage: coro <command> [options]",
    help="Run `coro <command> --help` for a command's options.",
)


@app.default
def _no_command(
    *tokens: Annotated[str, Parameter(allow_leading_hyphen=True, show=False)],
) -> NoReturn:
    """Refuse a bare invocation or an unknown command, naming the valid ones."""
    if tokens:
        sys.stderr.write(
            f"coro: unknown command {tokens[0]!r}. Expected one of: {', '.join(SUBCOMMANDS)}.\n\n"
        )
    sys.stderr.write(_USAGE)
    raise SystemExit(2)


@app.command(name="serve", help_flags=())
def _serve_command(*argv: Forwarded) -> None:
    """Start the HTTP transcription server."""
    serve(list(argv))


@app.command(name="run", help_flags=())
def _run_command(*argv: Forwarded) -> None:
    """Transcribe a local audio or video file without a server."""
    run(list(argv))


@app.command(name="bench", help_flags=())
def _bench_command(*argv: Forwarded) -> None:
    """Run the benchmark tooling (same as coro-bench)."""
    bench(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``coro`` command."""
    invoke(app, argv)


if __name__ == "__main__":
    main()
