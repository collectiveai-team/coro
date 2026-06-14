"""Vendored warmup audio and bench data assets.

Exposes path constants for the Warmup Audio Asset and other
static data files used by Server Warmup and Benchmark runs.
"""

from pathlib import Path

WARMUP_AUDIO_PATH: Path = Path(__file__).parent / "jfk.wav"
"""Path to the vendored JFK WAV (16 kHz mono, ~11 s)."""

SPANISH_REFERENCE_STMS: dict[str, Path] = {
    "RNE14-agosto-13": Path(__file__).parent / "spanish" / "RNE14-agosto-13.ref.stm",
}
"""Curated Spanish reference STMs (speaker-attributed) keyed by recording id.

Pair with the matching audio (not vendored; e.g. audios/RNE14-agosto-13.wav) via
``asr-diar-bench quality --audio <wav> --reference-stm <stm>`` or by dropping both
into a ``--clips-dir``.
"""
