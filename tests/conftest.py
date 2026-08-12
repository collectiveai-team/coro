"""Shared pytest fixtures.

Equivalents of the app/result helpers are duplicated in several older test
modules. New tests use these; migrating the existing copies is deliberately
left out of this change to avoid conflicting with the other branches in the
merge queue.
"""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from coro.app import create_app
from coro.bench import spanish
from coro.core.models import (
    ResponseSegment,
    TranscriptionResult,
    TranscriptItem,
    TranscriptWord,
)
from coro.runtime import RuntimeState
from coro.settings import ServerSettings

# (word, start, end, score, speaker) — two speakers plus one word the
# diarization timeline does not support.
DIARIZED_WORDS = [
    ("hola", 0.0, 0.5, 0.91, "1"),
    ("mundo", 0.5, 1.0, 0.83, "1"),
    ("si", 1.2, 1.6, 0.77, "2"),
    ("claro", 1.6, 2.0, 0.66, "-1"),
]

# The same words as a backend that reports no per-word probability sees them —
# onnx-genai and the onnx-asr text-only fallback both do. A ``None`` score must
# stay absent all the way to the wire rather than becoming a stubbed 1.0.
UNSCORED_WORDS = [(w, s, e, None, sp) for w, s, e, _, sp in DIARIZED_WORDS]


def make_result(words: list[tuple[str, float, float, float | None, str]]) -> TranscriptionResult:
    """Build a TranscriptionResult whose segments are same-speaker runs."""
    typed = [
        TranscriptWord(word=w, start=s, end=e, score=c, speaker=sp) for w, s, e, c, sp in words
    ]
    segments: list[ResponseSegment] = []
    for word in typed:
        if segments and segments[-1].speaker == word.speaker:
            segments[-1].words.append(word)
            segments[-1].end = word.end
            segments[-1].text = f"{segments[-1].text} {word.word}"
            continue
        segments.append(
            ResponseSegment(
                start=word.start,
                end=word.end,
                text=word.word,
                speaker=word.speaker,
                words=[word],
            )
        )
    return TranscriptionResult(
        segments=segments,
        word_segments=typed,
        transcript=[TranscriptItem(start=s.start, end=s.end, text=s.text) for s in segments],
        diarization=[],
        raw_words=[],
    )


class FakePipeline:
    """A pipeline returning a fixed result, for boundary tests."""

    def __init__(self, result: TranscriptionResult | None = None) -> None:
        self.result = result if result is not None else make_result(DIARIZED_WORDS)

    async def transcribe(self, audio, *, language=None, prompt=None):
        return self.result


def make_wav(*, frames: int = 1600, rate: int = 16000) -> bytes:
    """Return a valid, silent, mono 16-bit WAV payload."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    return buf.getvalue()


def make_app(pipeline: Any, settings: ServerSettings | None = None):
    """Build an app whose Singleton Runtime serves ``pipeline``."""
    application = create_app(settings or ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.pipeline = pipeline
    application.state.runtime = runtime
    return application


@pytest.fixture
def minimal_wav() -> bytes:
    """A valid, silent, mono 16-bit WAV payload."""
    return make_wav()


@pytest.fixture
def build_app() -> Callable[..., Any]:
    """Factory building an app whose Singleton Runtime serves a pipeline."""
    return make_app


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
