"""Shared test fixtures.

Equivalents of these helpers are duplicated in several older test modules. New
tests use these fixtures; migrating the existing copies is deliberately left
out of this change to avoid conflicting with the other branches in the merge
queue.
"""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import Callable
from typing import Any

import pytest

from coro.app import create_app
from coro.runtime import RuntimeState
from coro.settings import ServerSettings


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
