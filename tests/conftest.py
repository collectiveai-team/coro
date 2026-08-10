"""Shared pytest fixtures."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from coro.bench import spanish

SPANISH_CORPUS_ROWS: dict[str, list[dict]] = {
    "fleurs": [
        {
            "id": 101,
            "raw_transcription": "Hola, ¿cómo está el año?",
            "transcription": "hola como esta el ano",
            "audio": {"bytes": b"FAKE-1", "path": "101.wav"},
        },
        {
            "id": 102,
            "raw_transcription": "Buenos días a todos.",
            "transcription": "buenos dias a todos",
            "audio": {"bytes": b"FAKE-2", "path": "102.wav"},
        },
        {
            "id": 103,
            "raw_transcription": "   ",
            "transcription": "",
            "audio": {"bytes": b"FAKE-3", "path": "103.wav"},
        },
    ],
    "mls": [
        {
            "id": "10446_10446_000000",
            "transcript": "el camino era largo",
            "audio": {"bytes": b"FAKE-4", "path": "a.flac"},
        },
    ],
}


def write_silent_wav(dst: Path, seconds: float = 1.0) -> None:
    """Write a 16 kHz mono silent WAV, standing in for a transcoded corpus clip."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))


@pytest.fixture
def fake_spanish_corpus(monkeypatch):
    """Serve canned public-corpus rows and fake ffmpeg transcoding.

    Keeps Spanish Workload Set tests offline and free of an ffmpeg dependency
    while exercising the real materialisation, manifest and STM code paths.
    """
    monkeypatch.setattr(
        spanish,
        "resolve_shard_urls",
        lambda dataset, config, split: [f"https://example.invalid/{config}/{split}.parquet"],
    )

    def fake_iter(urls, *, limit, columns=None, timeout=60):
        key = "fleurs" if "es_419" in urls[0] else "mls"
        yield from SPANISH_CORPUS_ROWS[key][:limit]

    monkeypatch.setattr(spanish, "iter_parquet_rows", fake_iter)
    monkeypatch.setattr(
        spanish,
        "transcode_bytes_to_wav",
        lambda data, dst: write_silent_wav(dst),
    )


@pytest.fixture
def stub_server_handle() -> MagicMock:
    """Stand in for a Bench-Managed / Bench-Attached Server handle.

    Patch ``coro.bench.cli.build_server_handle`` with this so exercising
    ``coro.bench.cli.main`` never spawns a real server subprocess or blocks on
    ``/health`` polling.
    """
    handle = MagicMock()
    handle.__enter__.return_value = handle
    handle.base_url = "http://127.0.0.1:9999"
    handle.server_pid = 4242
    return handle
