"""The packaged CLI dispatches three named subcommands and nothing else.

The bare invocation used to mean "start the server", which left no room for a
second thing to do. Removing it is a breaking change (ADR 0017), so what a bare
``coro`` does now — refuse, and say what the alternatives are — is part of the
contract rather than an implementation detail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
import uvicorn
from starlette.testclient import TestClient
from support.factories import make_wav

from coro.backends.asr import factory as asr_factory
from coro.cli import SUBCOMMANDS, main
from coro.core.models import TranscriptToken


class _FakeASR:
    """In-memory ASR Adapter standing in for a model, so no model is ever loaded."""

    honours_prompt = False

    def __init__(self) -> None:
        self.windows = 0

    async def transcribe_pcm(
        self, pcm: bytes, *, language: str | None = None, prompt: str | None = None
    ) -> list[TranscriptToken]:
        self.windows += 1
        return [TranscriptToken(start=0.0, end=0.1, text=" hola.", probability=0.9)]


@dataclass
class _InProcessServer:
    """Stands in for uvicorn: starts the app in-process and sends it one transcription."""

    options: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    transcript: str = ""

    def run(self, app: Any, **options: Any) -> None:
        self.options = options
        with TestClient(app) as client:
            self.health = client.get("/health").json()
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", make_wav(frames=16000), "audio/wav")},
                data={"model": "whisper-1", "response_format": "json"},
            )
            self.transcript = response.json()["text"]


@pytest.fixture
def fake_asr(monkeypatch):
    """Serve the ASR Backend Adapter Factory an in-memory adapter."""
    adapter = _FakeASR()
    monkeypatch.setattr(asr_factory, "build_asr_adapter", lambda settings: adapter)
    return adapter


@pytest.fixture
def server(monkeypatch, fake_asr):
    fake = _InProcessServer()
    monkeypatch.setattr(uvicorn, "run", fake.run)
    return fake


def _exit_code(argv: list[str]) -> int | str | None:
    """Run the CLI and return its exit code, ``None`` when it returns normally."""
    try:
        main(argv)
    except SystemExit as exit_info:
        return exit_info.code
    return None


# MARK: Dispatch
@pytest.mark.parametrize(
    ("argv", "usage"),
    [
        (["serve", "--help"], "usage: coro serve"),
        (["run", "--help"], "Usage: coro run"),
        (["bench", "performance", "--help"], "Usage: coro-bench performance"),
    ],
    ids=SUBCOMMANDS,
)
def test_a_known_command_receives_the_arguments_after_it(argv, usage, capsys):
    assert _exit_code(argv) in (None, 0)

    assert usage in capsys.readouterr().out


def test_a_bare_invocation_is_refused(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "serve" in capsys.readouterr().err


def test_an_unknown_command_names_the_valid_ones(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["transcirbe"])

    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "transcirbe" in message
    assert all(command in message for command in SUBCOMMANDS)


def test_help_before_a_command_succeeds(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "coro <command>" in capsys.readouterr().out


def test_a_flag_alone_is_not_mistaken_for_a_command(capsys):
    """`coro --port 9000` was valid before; it must now fail rather than serve."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--port", "9000"])

    assert exit_info.value.code == 2
    assert "unknown command" in capsys.readouterr().err


# MARK: serve
def test_serve_starts_the_server_with_the_parsed_flags(server, fake_asr):
    main(["serve", "--port", "9123", "--host", "127.0.0.1", "--warmup", "disabled"])

    assert (server.options["port"], server.options["host"]) == (9123, "127.0.0.1")
    assert server.transcript == "hola."


def test_serve_with_warmup_disabled_reaches_the_model_only_for_requests(server, fake_asr):
    main(["serve", "--warmup", "disabled"])

    assert server.health["warmup_ready"] is True
    assert fake_asr.windows == 1


def test_serve_rejects_an_unknown_flag(server):
    with pytest.raises(SystemExit):
        main(["serve", "--not-a-real-flag", "1"])

    assert server.options == {}


# MARK: run
def test_run_transcribes_the_given_file(tmp_path, fake_asr):
    source = tmp_path / "input.wav"
    source.write_bytes(make_wav(frames=16000))
    output = tmp_path / "out.json"

    main(["run", str(source), "-o", str(output), "--warmup", "disabled"])

    assert json.loads(output.read_text())["text"] == "hola."


# MARK: bench
def test_bench_forwards_its_arguments(capsys):
    """The arguments after ``bench`` reach the coro-bench parser in order."""
    assert _exit_code(["bench", "performance", "--help"]) in (None, 0)

    out = capsys.readouterr().out
    assert "Usage: coro-bench performance" in out
    assert "--out-dir" in out
