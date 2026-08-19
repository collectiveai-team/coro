"""Canned public-corpus rows standing in for the Spanish Workload Set."""

from __future__ import annotations

import wave
from pathlib import Path

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
