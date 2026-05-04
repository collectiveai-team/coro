"""HTTP transport for sending audio to the ASR server."""

from __future__ import annotations

import json
import mimetypes
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
