"""Transcription Response factories and boundary stand-ins."""

from __future__ import annotations

import io
import wave
from typing import Any

from coro.app import create_app
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


def make_result(
    words: list[tuple[str, float, float, float | None, str]], *, duration: float = 0.0
) -> TranscriptionResult:
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
        duration=duration,
    )


class FakePipeline:
    """A pipeline returning a fixed result, for boundary tests."""

    def __init__(self, result: TranscriptionResult | None = None) -> None:
        self.result = result if result is not None else make_result(DIARIZED_WORDS)

    async def transcribe(self, audio, *, language=None, prompt=None):
        return self.result


def make_wav(*, frames: int = 1600, rate: int = 16000, pcm_prefix: bytes = b"") -> bytes:
    """Return a mono 16-bit WAV, padding the optional PCM prefix with silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm_prefix.ljust(frames * 2, b"\0"))
    return buf.getvalue()


def make_app(pipeline: Any, settings: ServerSettings | None = None):
    """Build an app whose Singleton Runtime serves ``pipeline``."""
    application = create_app(settings or ServerSettings())
    runtime = RuntimeState(asr_adapter=object())
    runtime.pipeline = pipeline
    application.state.runtime = runtime
    return application
