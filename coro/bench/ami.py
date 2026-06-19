"""AMI workload set resolution, auto-download, and reference STM materialization."""

from __future__ import annotations

import zipfile
from pathlib import Path

from coro.bench.stm import ami_meeting_to_stm, slice_stm_window
from coro.bench.utils.ami_audios_download import (
    download_annotations,
    download_meeting_audio,
)

AMI_GROUPS: dict[str, list[str]] = {
    "IB": ["IB4001", "IB4002", "IB4003", "IB4004", "IB4005"],
    "IN": [
        "IN1001",
        "IN1002",
        "IN1003",
        "IN1004",
        "IN1005",
        "IN1006",
        "IN1007",
        "IN1008",
    ],
    "ES": [
        "ES2002a",
        "ES2002b",
        "ES2003a",
        "ES2003b",
        "ES2004a",
        "ES2004b",
        "ES2005a",
        "ES2005b",
        "ES2006a",
        "ES2006b",
        "ES2007a",
        "ES2007b",
        "ES2008a",
        "ES2008b",
        "ES2009a",
        "ES2009b",
        "ES2010a",
        "ES2010b",
        "ES2011a",
        "ES2011b",
        "ES2012a",
        "ES2012b",
        "ES2013a",
        "ES2013b",
        "ES2014a",
        "ES2014b",
        "ES2015a",
        "ES2015b",
        "ES2016a",
        "ES2016b",
    ],
    "IS": [
        "IS1000a",
        "IS1000b",
        "IS1000c",
        "IS1000d",
        "IS1001a",
        "IS1001b",
        "IS1001c",
        "IS1001d",
        "IS1002b",
        "IS1002c",
        "IS1002d",
        "IS1003a",
        "IS1003b",
        "IS1003c",
        "IS1003d",
        "IS1004a",
        "IS1004b",
        "IS1004c",
        "IS1004d",
        "IS1005a",
        "IS1005b",
        "IS1005c",
        "IS1005d",
        "IS1006a",
        "IS1006b",
        "IS1006c",
        "IS1006d",
        "IS1007a",
        "IS1007b",
        "IS1007c",
        "IS1007d",
        "IS1008a",
        "IS1008b",
        "IS1008c",
        "IS1008d",
        "IS1009a",
        "IS1009b",
        "IS1009c",
        "IS1009d",
    ],
    "TS": [
        "TS3003a",
        "TS3003b",
        "TS3003c",
        "TS3003d",
        "TS3004a",
        "TS3004b",
        "TS3004c",
        "TS3004d",
        "TS3005a",
        "TS3005b",
        "TS3005c",
        "TS3005d",
        "TS3006a",
        "TS3006b",
        "TS3006c",
        "TS3006d",
        "TS3007a",
        "TS3007b",
        "TS3007c",
        "TS3007d",
        "TS3008a",
        "TS3008b",
        "TS3008c",
        "TS3008d",
        "TS3009a",
        "TS3009b",
        "TS3009c",
        "TS3009d",
        "TS3010a",
        "TS3010b",
        "TS3010c",
        "TS3010d",
        "TS3011a",
        "TS3011b",
        "TS3011c",
        "TS3011d",
        "TS3012a",
        "TS3012b",
        "TS3012c",
        "TS3012d",
    ],
    "EN": [
        "EN2001a",
        "EN2001b",
        "EN2002a",
        "EN2002b",
        "EN2002c",
        "EN2002d",
    ],
}

AMI_PRESETS: dict[str, list[str]] = {
    "sample": ["IB4001", "IN1001"],
    "eval": [m for group in ("ES", "IS") for m in AMI_GROUPS[group]],
    "full": [m for group in AMI_GROUPS for m in AMI_GROUPS[group]],
}


def resolve_workload_set(
    *,
    ami_meetings: list[str] | None = None,
    ami_groups: list[str] | None = None,
    ami_preset: str | None = None,
) -> list[str]:
    ami_meetings = ami_meetings or []
    ami_groups = ami_groups or []

    has_selector = bool(ami_meetings) or bool(ami_groups) or ami_preset is not None
    if not has_selector:
        ami_preset = "sample"

    seen: set[str] = set()
    result: list[str] = []

    def _add(meeting_id: str) -> None:
        if meeting_id not in seen:
            seen.add(meeting_id)
            result.append(meeting_id)

    for m in ami_meetings:
        _add(m)

    for g in ami_groups:
        for m in AMI_GROUPS.get(g, []):
            _add(m)

    if ami_preset:
        for m in AMI_PRESETS.get(ami_preset, []):
            _add(m)

    return result


def get_audio_path(ami_root: Path, meeting_id: str) -> Path:
    return ami_root / meeting_id / "audio" / f"{meeting_id}.Mix-Headset.wav"


def _annotations_extracted_marker(ami_root: Path) -> Path:
    return ami_root / ".ami_annotations_extracted"


def ensure_audio_and_annotations(
    meetings: list[str],
    ami_root: Path,
    *,
    no_download: bool = False,
) -> None:
    missing_audio: list[str] = []
    for meeting_id in meetings:
        audio_path = get_audio_path(ami_root, meeting_id)
        if not audio_path.exists():
            missing_audio.append(meeting_id)

    if no_download and missing_audio:
        raise RuntimeError(
            f"Missing audio for meetings (and --no-download was set): "
            f"{', '.join(sorted(missing_audio))}"
        )

    if not no_download:
        for meeting_id in meetings:
            download_meeting_audio(meeting_id, ami_root)

        if not _annotations_extracted_marker(ami_root).exists():
            zip_path = download_annotations(ami_root)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(ami_root)
            _annotations_extracted_marker(ami_root).touch()
    else:
        if not _annotations_extracted_marker(ami_root).exists():
            missing_annotations = sorted(meetings)
            raise RuntimeError(
                f"Missing annotations (and --no-download was set): {', '.join(missing_annotations)}"
            )


def clip_reference_stm(
    ami_root: Path,
    meeting_id: str,
    start: float,
    duration: float,
    *,
    recording_id: str | None = None,
) -> str:
    """Build a rebased reference STM for a short ``[start, start+duration)`` clip.

    Reuses the full-meeting AMI annotation conversion, then windows it so the
    references stay reliable on short, manually verifiable audio. Times are
    rebased to 0.0 to match a cut audio clip. ``recording_id`` overrides the STM
    session id (column 1) so it matches a hypothesis keyed by the clip stem.
    """
    full = ami_meeting_to_stm(ami_root, meeting_id)
    return slice_stm_window(
        full,
        start,
        start + duration,
        rebase=True,
        recording_id=recording_id,
    )


def materialize_reference_stms(
    meetings: list[str],
    ami_root: Path,
) -> None:
    stm_dir = ami_root / "stm"
    stm_dir.mkdir(parents=True, exist_ok=True)

    for meeting_id in meetings:
        stm_path = stm_dir / f"{meeting_id}.ref.stm"
        if stm_path.exists():
            continue
        stm_text = ami_meeting_to_stm(ami_root, meeting_id)
        stm_path.write_text(stm_text)
