"""Workload Set item model shared by the benchmark CLI and the orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path


class WorkloadItem(TypedDict):
    """One Workload Item: an audio file, its optional Reference STM and duration.

    ``audio_seconds`` starts at ``0.0`` and is filled in by the orchestrator
    from ``ffprobe`` before the item is first transcribed.
    """

    item_id: str
    audio_path: Path
    ref_stm_path: Path | None
    audio_seconds: float
