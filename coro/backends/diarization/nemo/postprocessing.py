"""Diarization Post-Processing Configuration resolution for NeMo Sortformer.

See ADR 0010. NeMo's own model card ships two dataset-optimized
post-processing presets (``dihard3-dev``, ``callhome-part1``); coro vendors
them verbatim and lets an operator select one, or supply a custom YAML in
the same schema, without coro computing or recommending any threshold
itself.
"""

from __future__ import annotations

from pathlib import Path

_PRESET_DIR = Path(__file__).parent / "postprocessing_presets"

# Preset name -> vendored filename. Keep in sync with the files actually
# present under postprocessing_presets/.
_PRESETS: dict[str, str] = {
    "dihard3-dev": "diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml",
    "callhome-part1": "diar_streaming_sortformer_4spk-v2_callhome-part1.yaml",
}


def resolve_postprocessing_yaml(value: str | None) -> str | None:
    """Resolve a Diarization Post-Processing Configuration value to a file path.

    ``None`` passes through unchanged, keeping NeMo's own unconfigured
    baseline. A recognized preset name resolves to its vendored file.
    Anything else is treated as a literal filesystem path and validated to
    exist, so an unresolvable value fails Server Startup Selection loudly
    rather than silently falling back to the baseline.
    """
    if value is None:
        return None
    if value in _PRESETS:
        return str(_PRESET_DIR / _PRESETS[value])
    path = Path(value)
    if not path.is_file():
        msg = (
            f"Diarization Post-Processing Configuration {value!r} is neither a "
            f"known preset ({sorted(_PRESETS)}) nor an existing file path."
        )
        raise ValueError(msg)
    return str(path)
