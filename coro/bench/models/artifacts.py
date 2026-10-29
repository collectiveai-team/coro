"""Shapes of the run artifacts the report builder reads back from disk.

``manifest.json``, ``quality/summary.json`` and ``performance/summary.json`` are
written by the orchestrator and read by ``bench.report``. Every key is optional
because the report tolerates artifacts from older runs; nested metric blocks
stay ``Mapping[str, object]`` and are read with type checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from coro.bench.models.response import HealthResponse


class WorkloadEntryJson(TypedDict, total=False):
    item_id: str
    audio_path: str
    ref_stm_path: str | None


class ManifestJson(TypedDict, total=False):
    """``manifest.json``."""

    subcommand: str
    timestamp: str
    git_sha: str
    cli_args: list[str]
    versions: dict[str, str]
    server_health: HealthResponse
    warmup: bool
    workload_set: list[WorkloadEntryJson]


class DiarizationJson(TypedDict, total=False):
    ref_speakers: int
    hyp_speakers: int
    degenerate: bool


class PerItemJson(TypedDict, total=False):
    """One ``per_item`` row of ``quality/summary.json``."""

    session_id: str
    audio_seconds: float | int
    error: object
    diarization: DiarizationJson | None
    diarization_only: bool | None
    cpwer: object
    orcwer: object
    dicpwer: object
    der: object
    wder: object
    wder_claimed: object
    abstention_rate: object
    normalized_cpwer: object
    normalized_orcwer: object
    normalized_dicpwer: object
    normalized_wder: object


class QualitySummaryJson(TypedDict, total=False):
    """``quality/summary.json``."""

    per_item: list[PerItemJson]
    combined: Mapping[str, object] | None


class PerRepJson(TypedDict, total=False):
    """One ``per_rep`` row of ``performance/summary.json``."""

    item_id: str
    rep: int
    audio_seconds: float | int
    wall_seconds: float | int
    transcription_throughput: float | int
    peak_pss_kb: object
    peak_pss_delta_kb: object
    peak_vram_mib: object
    peak_vram_delta_mib: object
    peak_gpu_util_pct: object
    peak_cpu_pct: object
    observed_hardware_profile: str
    time_to_first_delta_s: object


class PerformanceSummaryJson(TypedDict, total=False):
    """``performance/summary.json``."""

    per_rep: list[PerRepJson]
