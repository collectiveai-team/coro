"""Cycle 8: Audio module — aligned PCM chunking and edge cases.

Tests use small in-memory byte streams.  No real ffmpeg subprocess is invoked;
the aligned-chunking logic is tested in isolation using pre-built PCM bytes.
"""

from __future__ import annotations


from asr_diar_server.audio import SAMPLE_RATE, iter_aligned_pcm_chunks


# ---------------------------------------------------------------------------
# iter_aligned_pcm_chunks
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int) -> bytes:
    """Produce n_samples 16-bit little-endian samples (all zeros)."""
    return b"\x00\x00" * n_samples


def test_chunks_are_aligned_to_two_bytes():
    """Every yielded chunk has an even byte length (16-bit PCM alignment)."""
    pcm = _pcm_bytes(100)
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=30))
    for chunk in chunks:
        assert len(chunk) % 2 == 0, f"Unaligned chunk: {len(chunk)} bytes"


def test_total_bytes_preserved():
    """Total bytes across all chunks equals the input byte count."""
    pcm = _pcm_bytes(200)
    total_in = len(pcm)
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=64))
    total_out = sum(len(c) for c in chunks)
    assert total_out == total_in


def test_no_chunks_from_empty_input():
    """Empty input yields no chunks."""
    chunks = list(iter_aligned_pcm_chunks(iter([b""]), target_bytes=64))
    assert chunks == []


def test_single_sample_less_than_target_yielded_as_one_chunk():
    """Input smaller than target_bytes is yielded as a single chunk."""
    pcm = _pcm_bytes(4)  # 8 bytes < any reasonable target
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=64))
    assert len(chunks) == 1
    assert chunks[0] == pcm


def test_chunks_respect_target_bytes_upper_bound():
    """No chunk exceeds target_bytes."""
    pcm = _pcm_bytes(1000)
    target = 64
    chunks = list(iter_aligned_pcm_chunks(iter([pcm]), target_bytes=target))
    for chunk in chunks:
        assert len(chunk) <= target


def test_sample_rate_constant():
    """SAMPLE_RATE is 16000 Hz."""
    assert SAMPLE_RATE == 16000
