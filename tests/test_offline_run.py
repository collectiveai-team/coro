"""`coro run` transcribes a local file without a server and without destroying it.

The destructive case is the one worth guarding hardest: uploads are spooled
eagerly and both pipelines unlink what they are given when they finish, so an
owning AudioInput here would delete the user's input file. Everything else is
asserted the same way — through the CLI, against observable facts: was a
transcript produced, does the file still exist, was a socket opened.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from conftest import make_wav

from coro.cli import main
from coro.core.models import TranscriptToken


class _FakeASR:
    """ASR Adapter standing in for a model, so no model is ever loaded."""

    honours_prompt = False

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None]] = []

    async def transcribe_pcm(self, pcm, *, language=None, prompt=None):
        self.calls.append((language, prompt))
        return [TranscriptToken(start=0.0, end=0.1, text=" hola.", probability=0.9)]


@pytest.fixture
def audio_file(tmp_path):
    """A real, decodable WAV on disk — ffmpeg actually runs against it."""
    path = tmp_path / "input.wav"
    path.write_bytes(make_wav(frames=16000))
    return path


@pytest.fixture
def fake_asr():
    """Patch the ASR Backend Adapter Factory so no model is constructed."""
    adapter = _FakeASR()
    with patch("coro.backends.asr.factory.build_asr_adapter", autospec=True, return_value=adapter):
        yield adapter


@pytest.fixture
def no_network():
    """Fail loudly if anything opens an HTTP client."""

    def _refuse(*args, **kwargs):
        raise AssertionError("coro run contacted a server without being asked to")

    with patch("httpx.AsyncClient", new=_refuse):
        yield


def _run(*argv: str) -> None:
    main(["run", *argv])


# MARK: In-Process
def test_it_writes_a_transcript_for_a_local_file(audio_file, tmp_path, fake_asr, no_network):
    output = tmp_path / "out.json"
    _run(str(audio_file), "-o", str(output), "--warmup", "disabled")

    payload = json.loads(output.read_text())
    assert payload["text"] == "hola."


def test_it_leaves_the_input_file_in_place(audio_file, tmp_path, fake_asr, no_network):
    """Both pipelines unlink what they are given; the input must not be theirs."""
    before = audio_file.read_bytes()
    _run(str(audio_file), "-o", str(tmp_path / "out.json"), "--warmup", "disabled")

    assert audio_file.exists()
    assert audio_file.read_bytes() == before


def test_it_writes_to_stdout_when_no_output_is_given(audio_file, fake_asr, no_network, capsys):
    _run(str(audio_file), "--warmup", "disabled")

    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "hola."


def test_it_reports_the_configuration_that_produced_the_result(
    audio_file, tmp_path, fake_asr, no_network, capsys
):
    _run(str(audio_file), "-o", str(tmp_path / "out.json"), "--warmup", "disabled")

    report = capsys.readouterr().err
    assert "mode=in-process" in report
    assert "asr=onnx-asr:nemo-parakeet-tdt-0.6b-v3" in report


def test_it_forwards_the_language_hint(audio_file, tmp_path, fake_asr, no_network):
    _run(str(audio_file), "-o", str(tmp_path / "out.json"), "--language", "es")

    assert fake_asr.calls[0][0] == "es"


def test_it_accepts_server_settings_flags(audio_file, tmp_path, fake_asr, no_network, capsys):
    """The offline command must not need a second configuration vocabulary."""
    _run(str(audio_file), "-o", str(tmp_path / "out.json"), "--pipeline", "streaming")

    assert "pipeline=streaming" in capsys.readouterr().err


def test_it_rejects_an_unknown_flag(audio_file, fake_asr, no_network):
    with pytest.raises(SystemExit):
        _run(str(audio_file), "--not-a-real-flag", "1")


def test_it_rejects_a_missing_input_file(tmp_path, fake_asr, no_network, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _run(str(tmp_path / "absent.wav"))

    assert exit_info.value.code == 2
    assert "no such file" in capsys.readouterr().err


# MARK: Cache Reporting
def test_it_reports_cache_hits_and_misses(
    audio_file, tmp_path, fake_asr, no_network, undetectable_filesystem, capsys
):
    """Without this, there is no way to tell whether the cache is working."""
    cache_dir = str(tmp_path / "cache")
    arguments = (str(audio_file), "--asr-cache", "enabled", "--asr-cache-dir", cache_dir)

    _run(*arguments, "-o", str(tmp_path / "cold.json"))
    cold = capsys.readouterr().err

    _run(*arguments, "-o", str(tmp_path / "warm.json"))
    warm = capsys.readouterr().err

    assert "hits=0 misses=1" in cold
    assert "hits=1 misses=0" in warm


def test_a_fully_cached_run_is_byte_identical_to_the_cold_one(
    audio_file, tmp_path, fake_asr, no_network, undetectable_filesystem
):
    cache_dir = str(tmp_path / "cache")
    arguments = (str(audio_file), "--asr-cache", "enabled", "--asr-cache-dir", cache_dir)

    _run(*arguments, "-o", str(tmp_path / "cold.json"))
    _run(*arguments, "-o", str(tmp_path / "warm.json"))

    assert (tmp_path / "warm.json").read_text() == (tmp_path / "cold.json").read_text()


def test_a_fully_cached_run_never_builds_the_adapter(audio_file, tmp_path, undetectable_filesystem):
    """The point of the fast path is skipping the model load, not just inference."""
    cache_dir = str(tmp_path / "cache")
    arguments = (str(audio_file), "--asr-cache", "enabled", "--asr-cache-dir", cache_dir)

    with patch(
        "coro.backends.asr.factory.build_asr_adapter", autospec=True, return_value=_FakeASR()
    ) as build:
        _run(*arguments, "-o", str(tmp_path / "cold.json"))
        builds_after_cold_run = build.call_count
        _run(*arguments, "-o", str(tmp_path / "warm.json"))

    assert (builds_after_cold_run, build.call_count) == (1, 1)


# MARK: Parity With The Server
def test_its_output_matches_what_the_endpoint_would_return(audio_file, tmp_path, no_network):
    """Otherwise `coro run` is a fourth response shape nobody documented."""
    from starlette.testclient import TestClient

    from conftest import make_app
    from coro.offline import DEFAULT_RESPONSE_FORMAT
    from coro.pipelines.factory import build_pipeline
    from coro.settings import ServerSettings

    settings = ServerSettings(warmup="disabled", _env_file=None)
    pipeline = build_pipeline(settings, asr=_FakeASR())
    client = TestClient(make_app(pipeline, settings))
    with audio_file.open("rb") as handle:
        served = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("input.wav", handle, "audio/wav")},
            data={"response_format": DEFAULT_RESPONSE_FORMAT},
        )

    output = tmp_path / "out.json"
    with patch(
        "coro.backends.asr.factory.build_asr_adapter", autospec=True, return_value=_FakeASR()
    ):
        _run(str(audio_file), "-o", str(output), "--warmup", "disabled")

    assert json.loads(output.read_text()) == served.json()
