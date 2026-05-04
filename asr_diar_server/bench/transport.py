"""HTTP transport for sending audio to the ASR server."""

from __future__ import annotations

import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any


def transcribe_audio(
    base_url: str,
    audio_path: Path,
    *,
    timeout_seconds: float = 14400.0,
) -> dict[str, Any]:
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
    boundary = uuid.uuid4().hex
    parts = []

    parts.append(
        _form_field(boundary, "response_format", "diarized_json")
    )

    mime_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    filename = audio_path.name
    audio_bytes = audio_path.read_bytes()
    parts.append(
        _form_file(boundary, "file", filename, mime_type, audio_bytes)
    )

    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}"

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read())


def transcribe_audio_sse(
    base_url: str,
    audio_path: Path,
    *,
    timeout_seconds: float = 14400.0,
) -> tuple[dict[str, Any], float]:
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
    parts.append(
        _form_file(boundary, "file", filename, mime_type, audio_bytes)
    )

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
    done_payload: dict[str, Any] | None = None
    event_type: str = ""

    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
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


def _form_field(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n'
        f"\r\n"
        f"{value}\r\n"
    ).encode()


def _form_file(
    boundary: str,
    name: str,
    filename: str,
    mime_type: str,
    data: bytes,
) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n"
        f"\r\n"
    ).encode() + data + b"\r\n"
