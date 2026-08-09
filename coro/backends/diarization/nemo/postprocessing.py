"""Diarization Post-Processing Configuration resolution for NeMo Sortformer.

See ADR 0009. NeMo's own model card ships two dataset-optimized
post-processing presets (``dihard3-dev``, ``callhome-part1``); coro vendors
them verbatim and lets an operator select one, or supply a custom YAML in
the same schema, without coro computing or recommending any threshold
itself.

Each vendored preset records the scoring collar it was optimized against,
because the two are not interchangeable: zero-collar scoring rewards boundary
precision and near-zero padding, while collar-tolerant scoring rewards
generous padding and aggressive short-segment deletion. Callers that know
their scoring collar select with :func:`preset_for_collar` rather than
inheriting whichever default happens to apply.

This module also owns the Speaker-Count Post-Processing Gate — the single
place where both Diarization Flows decide whether the configured thresholds
should be applied at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_PRESET_DIR = Path(__file__).parent / "postprocessing_presets"


@dataclass(frozen=True)
class PostProcessingPreset:
    """A vendored NVIDIA post-processing parameter set and its provenance.

    Attributes:
        name: Selector value an operator sets.
        filename: Vendored YAML basename under ``postprocessing_presets/``.
        optimized_on: The corpus NVIDIA tuned the thresholds against.
        target_collar_s: The DER scoring collar the tuning targeted. Pairing a
            parameter set with a different collar is a measurement error, not
            a preference.
        source: Upstream provenance for the vendored file.

    """

    name: str
    filename: str
    optimized_on: str
    target_collar_s: float
    source: str


# Preset name -> vendored preset. Keep in sync with the files actually
# present under postprocessing_presets/.
_PRESETS: dict[str, PostProcessingPreset] = {
    "dihard3-dev": PostProcessingPreset(
        name="dihard3-dev",
        filename="diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml",
        optimized_on="DIHARD III dev split",
        target_collar_s=0.0,
        source="NVIDIA-NeMo/Speech (Apache-2.0), examples/speaker_tasks/diarization/"
        "conf/post_processing/",
    ),
    "callhome-part1": PostProcessingPreset(
        name="callhome-part1",
        filename="diar_streaming_sortformer_4spk-v2_callhome-part1.yaml",
        optimized_on="CALLHOME (NIST SRE 2000 Disc8), part1",
        target_collar_s=0.25,
        source="NVIDIA-NeMo/Speech (Apache-2.0), examples/speaker_tasks/diarization/"
        "conf/post_processing/",
    ),
}

# The Speaker-Count Post-Processing Gate's default ceiling.
#
# NVIDIA's own v2 model card reports that this post-processing consistently
# improves DER for four or fewer speakers and consistently *degrades* it at
# five or more (+0.26 to +0.66 absolute): short-segment deletion removes the
# brief, fragmentary evidence the model has for the additional speakers, so
# applying it unconditionally makes the worst case worse.
#
# NOTE: a 4-speaker Diarization Model Selection (every currently shipped
# Sortformer revision) emits a T x 4 activity matrix, so the estimate can
# never exceed 4 and this gate can never close. The mechanism exists so the
# behaviour is already correct when a >4-speaker model is selected; it is
# deliberately not dead code removal. See ADR 0009.
DEFAULT_MAX_SPEAKERS = 4

# Speaker-presence rule for the gate's estimate. Deliberately fixed rather
# than derived from the selected preset, so that the gate decision does not
# depend on the very thresholds it is gating.
_ESTIMATE_ONSET = 0.5
_ESTIMATE_MIN_SPEECH_S = 0.5


def resolve_postprocessing_yaml(value: str | None) -> str | None:
    """Resolve a Diarization Post-Processing Configuration value to a file path.

    ``None``, ``""`` and the explicit selector ``"none"`` all resolve to NeMo's
    own unconfigured baseline — the supported way to opt out now that the
    default is a tuned set. A recognized preset name resolves to its vendored
    file. Anything else is treated as a literal filesystem path and validated
    to exist, so an unresolvable value fails Server Startup Selection loudly
    rather than silently falling back to the baseline.
    """
    if value is None or value == "" or value == "none":
        return None
    if value in _PRESETS:
        return str(_PRESET_DIR / _PRESETS[value].filename)
    path = Path(value)
    if not path.is_file():
        msg = (
            f"Diarization Post-Processing Configuration {value!r} is neither a "
            f"known preset ({sorted(_PRESETS)}) nor an existing file path."
        )
        raise ValueError(msg)
    return str(path)


def preset_for_collar(collar_s: float) -> str:
    """Return the vendored preset name tuned for a DER scoring collar.

    Picks the preset whose ``target_collar_s`` is closest to the collar the
    caller actually scores at, so a benchmark lane cannot silently pair
    collar-tolerant thresholds with zero-collar scoring (or vice versa).
    """
    return min(_PRESETS, key=lambda name: abs(_PRESETS[name].target_collar_s - collar_s))


def preset_provenance() -> tuple[PostProcessingPreset, ...]:
    """Return the vendored presets with their provenance, for docs and reporting."""
    return tuple(_PRESETS[name] for name in sorted(_PRESETS))


def baseline_postprocessing_params() -> Any:
    """Return a fresh copy of NeMo's own unconfigured post-processing baseline.

    A new object every call: ``ts_vad_post_processing`` mutates the params it
    is handed when bypassing, so a shared instance would leak thresholds
    between calls.
    """
    from nemo.collections.asr.parts.mixins.diarization import load_postprocessing_from_yaml

    # NeMo accepts None to load default post-processing params; stub types str.
    return load_postprocessing_from_yaml(None)  # pyrefly: ignore[bad-argument-type]


def load_postprocessing_params(postprocessing_yaml: str | None) -> Any:
    """Load post-processing parameters from a resolved YAML path, or the baseline."""
    if postprocessing_yaml is None:
        return baseline_postprocessing_params()
    from nemo.collections.asr.parts.mixins.diarization import load_postprocessing_from_yaml

    return load_postprocessing_from_yaml(
        postprocessing_yaml=postprocessing_yaml,  # pyrefly: ignore[bad-argument-type]
    )


def _as_frames_by_speaker(preds: torch.Tensor) -> torch.Tensor:
    """Normalize raw Sortformer predictions to a 2-D ``(frames, speakers)`` tensor."""
    if preds.dim() == 3:
        preds = preds.squeeze(0)
    return preds.detach().cpu()


def estimate_speaker_count(
    preds: torch.Tensor,
    *,
    subsampling_factor: int = 8,
) -> int:
    """Estimate how many speakers the model actually found in the audio.

    A speaker counts as present when its activity exceeds a fixed onset
    threshold for at least a minimum total duration — enough to discount a
    handful of isolated noisy frames without needing the tuned thresholds
    whose application this estimate governs.

    Args:
        preds: Raw speaker-activity sigmoids, ``(frames, speakers)`` or
            ``(1, frames, speakers)``.
        subsampling_factor: Model frames per 10 ms unit, used to convert a
            frame count to seconds.

    Returns:
        The number of speakers judged present.

    """
    matrix = _as_frames_by_speaker(preds)
    if matrix.numel() == 0:
        return 0
    frame_seconds = subsampling_factor * 0.01
    active_frames = (matrix > _ESTIMATE_ONSET).sum(dim=0)
    return int((active_frames * frame_seconds >= _ESTIMATE_MIN_SPEECH_S).sum().item())


def postprocessing_gate_open(
    preds: torch.Tensor,
    *,
    subsampling_factor: int = 8,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> tuple[bool, int]:
    """Decide whether tuned post-processing should be applied to these predictions.

    Returns:
        ``(gate_open, estimated_speakers)``. The gate is open — meaning the
        configured thresholds apply — when the estimated speaker count is at
        or below ``max_speakers``. Above it, the tuned set is bypassed in
        favour of NeMo's plain baseline, because its short-segment deletion
        is reported to degrade DER precisely in that range.

    """
    estimated = estimate_speaker_count(preds, subsampling_factor=subsampling_factor)
    return estimated <= max_speakers, estimated


def segments_from_predictions(
    preds: torch.Tensor,
    *,
    n_spk: int,
    params: Any,
    subsampling_factor: int = 8,
) -> list[tuple[float, float, int]]:
    """Run NeMo's per-speaker VAD post-processing over raw predictions.

    The shared implementation behind both Diarization Flows, so batch and
    streaming cannot drift apart in how identical predictions become segments.

    Args:
        preds: Raw speaker-activity sigmoids for one recording.
        n_spk: Number of speaker channels to process.
        params: NeMo ``PostProcessingParams``. Consumed destructively by NeMo
            when bypassing; pass a fresh instance.
        subsampling_factor: Model frames per 10 ms unit.

    Returns:
        ``(start_seconds, end_seconds, speaker_index)`` tuples.

    """
    from nemo.collections.asr.models.sortformer_diar_models import ts_vad_post_processing

    matrix = _as_frames_by_speaker(preds)
    raw_segments: list[tuple[float, float, int]] = []
    for spk_id in range(n_spk):
        ts_mat = ts_vad_post_processing(
            matrix[:, spk_id],
            # NeMo consumes the PostProcessingParams dataclass; stub types OmegaConf.
            cfg_vad_params=params,  # pyrefly: ignore[bad-argument-type]
            unit_10ms_frame_count=subsampling_factor,
            bypass_postprocessing=False,
        )
        for start, end in ts_mat.detach().cpu().tolist():
            raw_segments.append((start, end, spk_id))
    return raw_segments


def apply_gated_postprocessing(
    preds: torch.Tensor,
    *,
    n_spk: int,
    postprocessing_yaml: str | None,
    subsampling_factor: int = 8,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> list[tuple[float, float, int]]:
    """Turn raw predictions into segments, honouring the speaker-count gate.

    When the gate is closed the configured thresholds are dropped in favour of
    NeMo's plain baseline for this recording only; the configuration itself is
    unchanged.
    """
    gate_open, estimated = postprocessing_gate_open(
        preds,
        subsampling_factor=subsampling_factor,
        max_speakers=max_speakers,
    )
    if gate_open:
        params = load_postprocessing_params(postprocessing_yaml)
    else:
        logger.info(
            "diarization postprocessing gate closed estimated_speakers=%d max_speakers=%d "
            "— falling back to the NeMo baseline for this recording",
            estimated,
            max_speakers,
        )
        params = baseline_postprocessing_params()
    return segments_from_predictions(
        preds,
        n_spk=n_spk,
        params=params,
        subsampling_factor=subsampling_factor,
    )
