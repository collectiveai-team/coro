"""Shared plumbing for coro's Cyclopts command lines: app defaults, invocation, output.

Every entry point keeps the exit codes it had under argparse: 0 after ``--help``,
2 for a usage error, and whatever the command itself returns or raises otherwise.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, NoReturn

import cyclopts
from rich.console import Console

# Plain consoles: no markup, highlighting or wrapping, so a command's product
# reaches stdout exactly as it was composed.
out_console = Console(markup=False, emoji=False, highlight=False, soft_wrap=True)
err_console = Console(stderr=True, markup=False, emoji=False, highlight=False, soft_wrap=True)

_BUILTIN_COMMANDS = (cyclopts.App.help_print, cyclopts.App.version_print)


def new_app(name: str, help: str | None = None, **kwargs: Any) -> cyclopts.App:  # noqa: A002
    """Build an App with the defaults every coro command line shares.

    Boolean flags get no ``--no-`` negation and there is no ``--version``, which
    is the surface the argparse parsers these apps replace had.

    Args:
        name: Program name shown in usage and errors.
        help: Description shown at the top of ``--help``.
        **kwargs: Further ``cyclopts.App`` options.

    Returns:
        The configured App.

    """
    return cyclopts.App(
        name=name,
        help=help,
        version_flags=(),
        help_format="plaintext",
        default_parameter=cyclopts.Parameter(negative=()),
        **kwargs,
    )


def invoke(app: cyclopts.App, argv: Sequence[str] | None = None) -> Any:
    """Parse ``argv`` with ``app`` and run the selected command.

    Args:
        app: The application to run.
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Whatever the selected command returns.

    Raises:
        SystemExit: Code 2 on a usage error, code 0 after printing help.

    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    command, bound, _ = app.parse_args(tokens)
    result = command(*bound.args, **bound.kwargs)
    if getattr(command, "__func__", None) in _BUILTIN_COMMANDS:
        raise SystemExit(0)
    return result


def fail(prog: str, message: str) -> NoReturn:
    """Report a usage error the way argparse did and exit with code 2.

    Args:
        prog: Program name to prefix the message with.
        message: What was wrong with the arguments.

    Raises:
        SystemExit: Always, with code 2.

    """
    err_console.out(f"{prog}: error: {message}")
    raise SystemExit(2)
