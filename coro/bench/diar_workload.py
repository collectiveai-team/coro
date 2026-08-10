"""Workload and reference resolution for the standalone diarization comparison.

Separated from ``eval_diarization_quality`` so that tool stays about running
adapters and scoring, while *what to run on* and *what to score against* live
here.

Two axes, independent:

**What to run on** — either whole AMI meetings resolved out of an AMI root, or a
directory of pre-cut clips (the layout ``make_ami_clip`` already writes:
``<clip>.wav`` beside ``<clip>.ref.stm``, with ``<clip>`` named
``<meeting>_<start>_<duration>``). Clips make a comparison runnable on a laptop
and let a workload be pinned exactly; whole meetings match how published
diarization results are reported.

**What to score against** — either the Reference STM this repo builds from AMI
manual annotation, or an external RTTM directory. The second exists because
published AMI diarization numbers are not all scored against the same reference:
NVIDIA's Sortformer cards use forced-alignment RTTMs
(``nttcslab-sp/diar-forced-alignment``) while most other published results use
``BUTSpeechFIT/AMI-diarization-setup``. Those two disagree by roughly 29% DER on
AMI, almost entirely in boundary placement, which is large enough to reverse a
model-selection verdict. A tool used to check a vendor figure has to be able to
score against the reference that figure was produced with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coro.bench.stm import rttm_to_stm, slice_stm_window

REF_SUFFIX = ".ref.stm"


@dataclass(frozen=True)
class DiarItem:
    """One audio file to diarize, and the reference to score it against.

    ``window`` is the ``(start, duration)`` the item covers within its source
    meeting, or ``None`` for a whole meeting. It is what lets an external
    whole-meeting RTTM be cut down to a clip.
    """

    item_id: str
    audio_path: Path
    ref_stm_path: Path
    meeting_id: str
    window: tuple[float, float] | None = None


def parse_clip_id(clip_id: str) -> tuple[str, tuple[float, float] | None]:
    """Split a clip stem into its meeting id and ``(start, duration)`` window.

    Clip stems are ``<meeting>_<start>_<duration>``. A stem that does not end in
    two numeric fields is treated as a whole meeting, so a directory of
    full-length recordings still resolves.
    """
    parts = clip_id.rsplit("_", 2)
    if len(parts) == 3:
        try:
            return parts[0], (float(parts[1]), float(parts[2]))
        except ValueError:
            pass
    return clip_id, None


def items_from_clips_dir(clips_dir: Path) -> list[DiarItem]:
    """Resolve every ``<clip>.wav`` in a clips directory into a DiarItem.

    Raises when a clip has no sibling Reference STM, rather than silently
    scoring a subset — a partially-resolved workload produces a combined DER
    that looks valid and is not comparable to anything.
    """
    items: list[DiarItem] = []
    missing: list[str] = []
    for wav_path in sorted(clips_dir.glob("*.wav")):
        clip_id = wav_path.stem
        ref_path = clips_dir / f"{clip_id}{REF_SUFFIX}"
        if not ref_path.exists():
            missing.append(clip_id)
            continue
        meeting_id, window = parse_clip_id(clip_id)
        items.append(
            DiarItem(
                item_id=clip_id,
                audio_path=wav_path,
                ref_stm_path=ref_path,
                meeting_id=meeting_id,
                window=window,
            )
        )
    if missing:
        msg = f"Clips missing a {REF_SUFFIX} reference: {', '.join(missing)}"
        raise FileNotFoundError(msg)
    if not items:
        msg = f"No .wav clips found in {clips_dir}"
        raise FileNotFoundError(msg)
    return items


def items_from_meetings(meetings: list[str], ami_root: Path) -> list[DiarItem]:
    """Resolve whole AMI meetings into DiarItems using the AMI root layout."""
    return [
        DiarItem(
            item_id=meeting_id,
            audio_path=ami_root / meeting_id / "audio" / f"{meeting_id}.Mix-Headset.wav",
            ref_stm_path=ami_root / "stm" / f"{meeting_id}{REF_SUFFIX}",
            meeting_id=meeting_id,
        )
        for meeting_id in meetings
    ]


def _find_rttm(rttm_dir: Path, meeting_id: str) -> Path | None:
    """Locate a meeting's RTTM, searching one level of split subdirectories.

    Published RTTM sets ship either flat or partitioned into ``train``/``dev``/
    ``test``; both layouts resolve without the caller having to know which.
    """
    direct = rttm_dir / f"{meeting_id}.rttm"
    if direct.exists():
        return direct
    matches = sorted(rttm_dir.glob(f"*/{meeting_id}.rttm"))
    return matches[0] if matches else None


def materialize_rttm_references(
    items: list[DiarItem],
    rttm_dir: Path,
    out_dir: Path,
) -> list[DiarItem]:
    """Rewrite each item to score against an external RTTM instead of its STM.

    Whole-meeting RTTMs are windowed and rebased per item so a clip is scored
    only against the reference turns its audio actually contains — the same
    treatment ``clip_reference_stm`` gives the built-in reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rebased: list[DiarItem] = []
    missing: list[str] = []

    for item in items:
        rttm_path = _find_rttm(rttm_dir, item.meeting_id)
        if rttm_path is None:
            missing.append(item.meeting_id)
            continue

        full = rttm_to_stm(rttm_path.read_text(), item.meeting_id)
        if item.window is None:
            text = rttm_to_stm(rttm_path.read_text(), item.item_id)
        else:
            start, duration = item.window
            text = slice_stm_window(
                full, start, start + duration, rebase=True, recording_id=item.item_id
            )

        ref_path = out_dir / f"{item.item_id}{REF_SUFFIX}"
        ref_path.write_text(text)
        rebased.append(
            DiarItem(
                item_id=item.item_id,
                audio_path=item.audio_path,
                ref_stm_path=ref_path,
                meeting_id=item.meeting_id,
                window=item.window,
            )
        )

    if missing:
        msg = f"No RTTM under {rttm_dir} for: {', '.join(sorted(set(missing)))}"
        raise FileNotFoundError(msg)
    return rebased
