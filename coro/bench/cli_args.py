"""Argument parsing and validation for the coro-bench CLI.

Kept separate from the run orchestration in :mod:`coro.bench.cli` so the flag
surface can be read and validated on its own; the flags themselves are the
fields of :class:`coro.bench.cli_flags.BenchArgs`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Annotated, NoReturn

from cyclopts import Parameter

from coro.bench.cli_flags import BenchArgs
from coro.bench.quarantine import CircularReferenceError, assert_scorable_reference
from coro.cli_support import fail, invoke, new_app

__all__ = ["BenchArgs", "parse_args"]

PROG = "coro-bench"
SUBCOMMANDS = ("quality", "performance", "all")

_MANAGED_FLAGS = (
    "--server-asr-backend",
    "--server-asr-model",
    "--server-diar-backend",
    "--server-diar-model",
    "--server-pipeline",
    "--server-port",
    "--no-diarization",
)

_ATTACHED_FLAGS = ("server_pid", "server_match")

# Subcommands that sample the Server Process Tree and therefore need a real PID.
_SAMPLING_SUBCOMMANDS = frozenset({"performance", "all"})


def _error(message: str) -> NoReturn:
    fail(PROG, message)


def _given_flags(tokens: Sequence[str]) -> set[str]:
    """Return the ``--flag`` names present in ``tokens``, with any ``=value`` removed."""
    return {token.split("=", 1)[0] for token in tokens if token.startswith("--")}


def _validate_reference_args(args: BenchArgs) -> None:
    """Reject quarantined references and conflicting workload-set selectors."""
    if args.reference_stm is not None:
        try:
            assert_scorable_reference(args.reference_stm)
        except CircularReferenceError as exc:
            _error(str(exc))

    if args.spanish_preset is not None and args.clips_dir is not None:
        _error("--spanish-preset is mutually exclusive with --clips-dir.")

    if args.spanish_fetch_plan and args.spanish_preset is None:
        _error("--spanish-fetch-plan requires --spanish-preset.")


def _validate_transport_args(args: BenchArgs) -> None:
    """Reject transport selections that cannot be honoured.

    ``--deepgram`` is deliberately allowed on ``quality``, unlike ``--stream``:
    quality is the subcommand that computes WDER, and the Deepgram endpoint is
    the only surface carrying the per-word speakers WDER is meant to score.
    """
    if args.stream and args.subcommand == "quality":
        _error("--stream is not allowed for the 'quality' subcommand.")

    if args.stream and args.deepgram:
        _error(
            "--stream and --deepgram select different endpoints; /v1/listen is not an SSE surface."
        )


def _validate_args(args: BenchArgs) -> None:
    _validate_transport_args(args)

    has_attached = args.server_url is not None
    has_managed_explicit = not _given_flags(args.cli_args).isdisjoint(_MANAGED_FLAGS)

    if has_attached and has_managed_explicit:
        _error(
            "--server-url is mutually exclusive with bench-managed server flags "
            "(--server-asr-backend, --server-asr-model, --server-diar-backend, "
            "--server-diar-model, --server-pipeline, --server-port, --no-diarization)."
        )

    if not has_attached:
        for flag in _ATTACHED_FLAGS:
            if getattr(args, flag) is not None:
                _error(
                    f"--{flag.replace('_', '-')} only applies to a bench-attached server; "
                    "a bench-managed server reports its own PID. Pass --server-url to "
                    "attach to an already-running server."
                )

    if args.warmup_audio is not None:
        args.warmup = True

    if args.no_diarization:
        args.server_diar_backend = "none"

    if args.reference_stm is not None and args.audio is None:
        _error("--reference-stm requires --audio.")

    if args.audio is not None and args.reference_stm is None and args.subcommand == "quality":
        _error("--audio without --reference-stm is not allowed for the 'quality' subcommand.")

    _validate_reference_args(args)


def _subcommand(name: str) -> Callable[..., BenchArgs]:
    def command(args: BenchArgs | None = None) -> BenchArgs:
        selected = args if args is not None else BenchArgs()
        selected.subcommand = name
        return selected

    return command


def _missing_subcommand(*tokens: Annotated[str, Parameter(show=False)]) -> NoReturn:
    if tokens:
        choices = ", ".join(repr(name) for name in SUBCOMMANDS)
        _error(f"argument subcommand: invalid choice: {tokens[0]!r} (choose from {choices})")
    _error("the following arguments are required: subcommand")


_app = new_app(PROG, help="Benchmark an ASR HTTP endpoint.")
_app.default(_missing_subcommand)
for _name in SUBCOMMANDS:
    _app.command(_subcommand(_name), name=_name)


def parse_args(argv: Sequence[str] | None = None) -> BenchArgs:
    """Parse and validate coro-bench arguments.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        The validated arguments, with every default filled in.

    Raises:
        SystemExit: Code 2 on invalid arguments, code 0 after ``--help``.

    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    args: BenchArgs = invoke(_app, tokens)
    # Recorded verbatim in the run manifest so a result can be re-run as issued.
    args.cli_args = tokens
    _validate_args(args)
    return args
