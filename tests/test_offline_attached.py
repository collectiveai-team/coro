"""`coro run --server-url` attaches to a running server, and only then.

Attaching is worth having when a server already holds a warm model, but it must
never be inferred. A command that behaved differently depending on whether
something happened to be listening on a port would be unpredictable, and an
attached run is governed by the server's own configuration rather than the flags
typed locally — so it reports that configuration instead of the local one.

The "server" here is a real app served over an in-memory transport, so the
request path, the multipart body and the response schema are all exercised
without a socket.
"""

from __future__ import annotations

import json

import pytest
from support.factories import make_app, make_wav
from httpx import ASGITransport, AsyncClient

import coro.offline
from coro.core.models import ResponseSegment, TranscriptionResult, TranscriptItem
from coro.offline import transcribe_attached


class _FakePipeline:
    """Pipeline returning a fixed result, so the server loads no model."""

    def __init__(self) -> None:
        self.uploads: list[int] = []

    async def transcribe(self, audio, *, language=None, prompt=None):
        self.uploads.append(audio.size)
        return TranscriptionResult(
            segments=[ResponseSegment(start=0.0, end=1.0, text="remote.", speaker="1")],
            transcript=[TranscriptItem(start=0.0, end=1.0, text="remote.")],
        )

    def stream(self, audio, *, language=None, prompt=None):
        raise NotImplementedError


def _attach_to(application, monkeypatch) -> None:
    """Point attached runs at an ASGI app instead of a live socket."""
    monkeypatch.setattr(
        coro.offline,
        "build_http_client",
        lambda: AsyncClient(transport=ASGITransport(app=application)),
    )


@pytest.fixture
def server(monkeypatch):
    """A running server, addressable at any URL, returning a fixed transcript."""
    pipeline = _FakePipeline()
    _attach_to(make_app(pipeline), monkeypatch)
    return pipeline


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(make_wav(frames=16000))
    return path


@pytest.mark.asyncio
async def test_it_returns_the_servers_transcript(server, audio_file):
    body, _ = await transcribe_attached(str(audio_file), server_url="http://server")

    payload = json.loads(body)
    assert payload["text"] == "remote."
    assert payload["segments"][0]["speaker"] == "1"


@pytest.mark.asyncio
async def test_it_uploads_the_file_rather_than_transcribing_it_locally(server, audio_file):
    await transcribe_attached(str(audio_file), server_url="http://server")

    assert server.uploads == [audio_file.stat().st_size]


@pytest.mark.asyncio
async def test_it_leaves_the_input_file_in_place(server, audio_file):
    before = audio_file.read_bytes()
    await transcribe_attached(str(audio_file), server_url="http://server")

    assert audio_file.exists()
    assert audio_file.read_bytes() == before


@pytest.mark.asyncio
async def test_it_reports_the_servers_configuration_not_the_local_one(server, audio_file):
    """The local flags did not produce this result, so they must not be reported."""
    _, report = await transcribe_attached(str(audio_file), server_url="http://server/")

    assert report.mode == "attached"
    assert report.server_url == "http://server"
    assert report.pipeline == "full-memory"
    assert "server=http://server" in report.summary()


@pytest.mark.asyncio
async def test_a_server_error_is_surfaced(audio_file, monkeypatch):
    async def _always_500(scope, receive, send):
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"boom"})

    _attach_to(_always_500, monkeypatch)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await transcribe_attached(str(audio_file), server_url="http://server")


def test_the_cli_attaches_only_when_a_server_url_is_given(server, audio_file, tmp_path):
    """Without the flag nothing is contacted; the fixture would record an upload."""
    from unittest.mock import patch

    from coro.cli import main
    from coro.core.models import TranscriptToken

    class _FakeASR:
        honours_prompt = False

        async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
            return [TranscriptToken(start=0.0, end=0.1, text=" local.", probability=0.9)]

    with patch(
        "coro.backends.asr.factory.build_asr_adapter", autospec=True, return_value=_FakeASR()
    ):
        main(["run", str(audio_file), "-o", str(tmp_path / "out.json")])

    assert server.uploads == []


def test_the_cli_attaches_when_a_server_url_is_given(server, audio_file, tmp_path, capsys):
    from coro.cli import main

    main(["run", str(audio_file), "--server-url", "http://server", "-o", str(tmp_path / "o.json")])

    assert server.uploads == [audio_file.stat().st_size]
    assert "mode=attached" in capsys.readouterr().err


def test_settings_flags_are_reported_as_ignored_when_attached(server, audio_file, tmp_path, capsys):
    """Silently ignoring them would let someone believe they ran a backend they did not."""
    from coro.cli import main

    main(
        [
            "run",
            str(audio_file),
            "--server-url",
            "http://server",
            "--backend-asr",
            "faster-whisper",
            "-o",
            str(tmp_path / "o.json"),
        ]
    )

    message = capsys.readouterr().err
    assert "ignoring" in message
    assert "faster-whisper" in message
