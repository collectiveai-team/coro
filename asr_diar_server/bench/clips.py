"""Resolve a directory of (audio, reference STM) pairs into workload items.

Used for short, reliable-reference benchmarks (AMI clips via make_ami_clip, or
curated non-AMI audio such as the Spanish RNE14 reference). Each ``<stem>.wav``
is paired with a sibling ``<stem>.ref.stm`` when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def resolve_clip_items(clips_dir: Path) -> list[dict[str, Any]]:
    """Return workload items for every audio file in ``clips_dir``.

    Each item pairs ``<stem>.<audio-ext>`` with ``<stem>.ref.stm`` (None when the
    reference is absent). Items are sorted by stem for deterministic ordering.
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for audio in sorted(clips_dir.iterdir() if clips_dir.is_dir() else []):
        if audio.suffix.lower() not in _AUDIO_SUFFIXES or audio.stem in seen:
            continue
        seen.add(audio.stem)
        ref = clips_dir / f"{audio.stem}.ref.stm"
        items.append(
            {
                "item_id": audio.stem,
                "audio_path": audio,
                "ref_stm_path": ref if ref.exists() else None,
                "audio_seconds": 0.0,
            }
        )
    return items
