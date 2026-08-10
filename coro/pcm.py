"""Live PCM ingest for streaming transcription.

HTTP uploads reach the pipeline through ffmpeg, which sniffs the container and
emits canonical PCM. A live socket has no container to sniff: the client
*declares* its encoding and sample rate up front and then sends bare samples.
This module validates that declaration and converts the stream.

Canonical form is 16 kHz, 16-bit little-endian, mono — see :mod:`coro.audio`.
"""

from __future__ import annotations

import numpy as np
import soxr

from coro.audio import BYTES_PER_SAMPLE, SAMPLE_RATE

LINEAR16 = "linear16"
"""The only encoding accepted on a live socket: bare 16-bit LE samples."""

SUPPORTED_ENCODINGS = frozenset({LINEAR16, ""})

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000


class UnsupportedAudioFormat(ValueError):
    """The client declared an encoding or rate this server cannot ingest."""


def validate_format(encoding: str | None, sample_rate: int | None, channels: int | None) -> None:
    """Reject a declared audio format that cannot be ingested.

    Raised before any audio is accepted, so a misconfigured client fails at
    connection time with a clear reason rather than after streaming a minute of
    audio that silently decoded to noise.

    Args:
        encoding: The client's ``encoding`` parameter.
        sample_rate: The client's ``sample_rate`` parameter.
        channels: The client's ``channels`` parameter.

    Raises:
        UnsupportedAudioFormat: If any of the three cannot be honoured.

    """
    normalized = (encoding or "").strip().lower()
    if normalized not in SUPPORTED_ENCODINGS:
        raise UnsupportedAudioFormat(f"encoding '{normalized}' is not supported; use '{LINEAR16}'")
    if sample_rate is not None and not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise UnsupportedAudioFormat(
            f"sample_rate {sample_rate} is outside the supported "
            f"{MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz range"
        )
    if channels is not None and channels != 1:
        raise UnsupportedAudioFormat(f"channels {channels} is not supported; audio must be mono")


class PcmStreamConverter:
    """Convert a live ``linear16`` stream to canonical 16 kHz mono PCM.

    Stateful on purpose. A resampler restarted per chunk produces a
    discontinuity at every boundary, which a chunked socket stream would hit
    many times a second; :class:`soxr.ResampleStream` carries its filter state
    across calls instead. It also absorbs odd byte counts, since a socket frame
    can split a 16-bit sample down the middle.
    """

    def __init__(self, *, source_rate: int) -> None:
        self._source_rate = source_rate
        self._remainder = b""
        self._stream = (
            None
            if source_rate == SAMPLE_RATE
            else soxr.ResampleStream(source_rate, SAMPLE_RATE, 1, dtype="int16", quality="HQ")
        )

    @property
    def resampling(self) -> bool:
        """True when incoming audio is not already at the canonical rate."""
        return self._stream is not None

    def push(self, data: bytes) -> bytes:
        """Convert one inbound frame, holding back any partial trailing sample."""
        return self._convert(data, last=False)

    def flush(self) -> bytes:
        """Drain the resampler's tail at end of stream."""
        return self._convert(b"", last=True)

    def _convert(self, data: bytes, *, last: bool) -> bytes:
        buffer = self._remainder + data
        usable = len(buffer) - (len(buffer) % BYTES_PER_SAMPLE)
        self._remainder = buffer[usable:]
        payload = buffer[:usable]
        if self._stream is None:
            return payload
        samples = np.frombuffer(payload, dtype="<i2")
        converted = self._stream.resample_chunk(samples, last=last)
        return np.asarray(converted, dtype="<i2").tobytes()
