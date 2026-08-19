"""The packaged CLI dispatches three named subcommands and nothing else.

The bare invocation used to mean "start the server", which left no room for a
second thing to do. Removing it is a breaking change (ADR 0017), so what a bare
``coro`` does now — refuse, and say what the alternatives are — is part of the
contract rather than an implementation detail.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from coro.cli import SUBCOMMANDS, main, split_subcommand


# MARK: Dispatch
@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_a_known_command_is_split_from_its_arguments(command):
    assert split_subcommand([command, "--flag", "value"]) == (command, ["--flag", "value"])


def test_a_bare_invocation_is_refused(capsys):
    with pytest.raises(SystemExit) as exit_info:
        split_subcommand([])

    assert exit_info.value.code == 2
    assert "serve" in capsys.readouterr().err


def test_an_unknown_command_names_the_valid_ones(capsys):
    with pytest.raises(SystemExit) as exit_info:
        split_subcommand(["transcirbe"])

    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "transcirbe" in message
    assert all(command in message for command in SUBCOMMANDS)


def test_help_before_a_command_succeeds(capsys):
    with pytest.raises(SystemExit) as exit_info:
        split_subcommand(["--help"])

    assert exit_info.value.code == 0
    assert "coro <command>" in capsys.readouterr().out


def test_a_flag_alone_is_not_mistaken_for_a_command(capsys):
    """`coro --port 9000` was valid before; it must now fail rather than serve."""
    with pytest.raises(SystemExit) as exit_info:
        split_subcommand(["--port", "9000"])

    assert exit_info.value.code == 2
    assert "unknown command" in capsys.readouterr().err


# MARK: serve
def test_serve_starts_the_server_with_the_parsed_flags():
    with (
        patch("uvicorn.run", autospec=True) as run,
        patch("coro.app.create_app", autospec=True) as create_app,
    ):
        main(["serve", "--port", "9123", "--host", "127.0.0.1", "--warmup", "disabled"])

    settings = create_app.call_args.args[0]
    assert (settings.port, settings.host, settings.warmup) == (9123, "127.0.0.1", "disabled")
    assert run.call_args.kwargs["port"] == 9123


def test_serve_rejects_an_unknown_flag():
    with pytest.raises(SystemExit), patch("uvicorn.run", autospec=True):
        main(["serve", "--not-a-real-flag", "1"])


# MARK: bench
def test_bench_forwards_its_arguments():
    with patch("coro.bench.cli.main", autospec=True) as bench_main:
        main(["bench", "performance", "--out-dir", "/tmp/x"])

    bench_main.assert_called_once_with(["performance", "--out-dir", "/tmp/x"])
