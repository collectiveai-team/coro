"""HTTP transport for sending audio to the ASR server."""

from __future__ import annotations

import json
import mimetypes
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coro.bench.errors import ServerUnreachableError

TranscribeFn = Callable[[str, Path], tuple[Any, float | None]]
"""A transport: takes ``(base_url, audio_path)``, returns ``(response, ttft)``.

``ttft`` is the time to first delta, which only the SSE transport can measure;
the others report ``None`` rather than a fabricated zero.
"""


def _is_connection_refused(exc: BaseException) -> bool:
    """Return True if exc represents a refused/unreachable TCP connection.

    Covers ConnectionRefusedError directly as well as urllib.error.URLError
    wrapping it (which is what urlopen typically raises in practice).
    """
    import urllib.error

    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ConnectionRefusedError):
            return True
        if isinstance(reason, OSError) and reason.errno in (
            111,  # ECONNREFUSED on Linux
            61,  # ECONNREFUSED on macOS
        ):
            return True
    return False


def transcribe_audio(
    base_url: str,
    audio_path: Path,
    *,
    timeout_seconds: float = 14400.0,
) -> Any:
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
    boundary = uuid.uuid4().hex
    parts = []

    parts.append(_form_field(boundary, "response_format", "diarized_json"))

    mime_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    filename = audio_path.name
    audio_bytes = audio_path.read_bytes()
    parts.append(_form_file(boundary, "file", filename, mime_type, audio_bytes))

    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}"

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        if _is_connection_refused(exc):
            raise ServerUnreachableError(base_url, cause=exc) from exc
        raise


DEEPGRAM_QUERY = "diarize=true&utterances=true"
"""Not configurable, and deliberately so.

Both default to ``false`` at the endpoint, as they do at Deepgram, and an
undiarized response is byte-identical to one where diarization abstained on
every word — Deepgram omits ``speaker`` in both cases. A caller able to pass
``diarize=false`` could therefore produce a run that scores every word as
abstention and still reports a WDER, with nothing downstream able to detect it.
"""


def transcribe_audio_deepgram(
    base_url: str,
    audio_path: Path,
    *,
    timeout_seconds: float = 14400.0,
) -> Any:
    """POST the audio to the Deepgram-native ``/v1/listen`` endpoint.

    Deepgram takes a **raw audio body**, not multipart, so this cannot reuse
    the OpenAI request builder above. It is the only wire surface coro serves
    that carries per-word speaker labels; ``diarized_json`` has no word field,
    so a benchmark run over the OpenAI endpoint can only ever score the
    segment-level speaker summary.

    Diarization is always requested; see :data:`DEEPGRAM_QUERY`.
    """
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/listen?{DEEPGRAM_QUERY}"
    mime_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    req = urllib.request.Request(
        url,
        data=audio_path.read_bytes(),
        headers={"Content-Type": mime_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        if _is_connection_refused(exc):
            raise ServerUnreachableError(base_url, cause=exc) from exc
        raise


def transcribe_audio_sse(
    base_url: str,
    audio_path: Path,
    *,
    timeout_seconds: float = 14400.0,
) -> tuple[Any, float]:
    """POST with stream=true, parse SSE events.

    Returns (diarized_json_from_done_event, time_to_first_delta_s).
    """
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
    boundary = uuid.uuid4().hex
    parts = []

    parts.append(_form_field(boundary, "stream", "true"))

    mime_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    filename = audio_path.name
    audio_bytes = audio_path.read_bytes()
    parts.append(_form_file(boundary, "file", filename, mime_type, audio_bytes))

    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}"

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    start_time = time.monotonic()
    first_delta_time: float | None = None
    done_payload: Any | None = None
    event_type: str = ""

    try:
        resp_cm = urllib.request.urlopen(req, timeout=timeout_seconds)
    except Exception as exc:
        if _is_connection_refused(exc):
            raise ServerUnreachableError(base_url, cause=exc) from exc
        raise

    with resp_cm as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\n\r")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event_type == "transcript.text.delta" and first_delta_time is None:
                    first_delta_time = time.monotonic() - start_time
                elif event_type == "transcript.text.done":
                    done_payload = json.loads(data["text"])

    if first_delta_time is None:
        first_delta_time = time.monotonic() - start_time
    if done_payload is None:
        raise RuntimeError("SSE stream ended without a transcript.text.done event")

    return done_payload, first_delta_time


def select_transport(*, stream: bool = False, deepgram: bool = False) -> TranscribeFn:
    """Return the transport a run's configuration asks for.

    The workloads differ in what they measure but not in how they choose an
    endpoint, so the choice lives here once rather than as an ``if`` repeated at
    every call site. Repeating it is what left the quality workload — the one
    that computes WDER — with no choice at all, pinned to the OpenAI endpoint
    whose ``diarized_json`` carries no per-word speakers.

    Every transport returns ``(response, time_to_first_delta)`` so callers need
    no branch of their own; only the SSE transport can measure that latency, and
    the others report ``None`` rather than a fabricated zero.

    Args:
        stream: Use the OpenAI SSE path and record time-to-first-delta.
        deepgram: Use the Deepgram-native endpoint, the only wire surface that
            carries per-word speaker labels.

    Raises:
        ValueError: If both are requested. The Deepgram endpoint is not an SSE
            surface, so the combination has no meaning and would otherwise
            resolve by silent precedence.

    """
    if stream and deepgram:
        raise ValueError(
            "stream and deepgram select different endpoints; /v1/listen is not an SSE surface."
        )
    if deepgram:
        return lambda base_url, audio_path: (
            transcribe_audio_deepgram(base_url, audio_path),
            None,
        )
    if stream:
        return transcribe_audio_sse
    return lambda base_url, audio_path: (transcribe_audio(base_url, audio_path), None)


def _form_field(boundary: str, name: str, value: str) -> bytes:
    return (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
    ).encode()


def _form_file(
    boundary: str,
    name: str,
    filename: str,
    mime_type: str,
    data: bytes,
) -> bytes:
    return (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n"
            f"\r\n"
        ).encode()
        + data
        + b"\r\n"
    )
