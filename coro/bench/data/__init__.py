"""Vendored warmup audio and bench data assets.

Exposes path constants for the Warmup Audio Asset and other
static data files used by Server Warmup and Benchmark runs.
"""

from pathlib import Path

WARMUP_AUDIO_PATH: Path = Path(__file__).parent / "jfk.wav"
"""Path to the vendored JFK WAV (16 kHz mono, ~11 s)."""
