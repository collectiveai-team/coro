"""StreamingDiarizer finalize and post-processing, with a mocked Sortformer model.

No real NeMo model is loaded; the mocks come from this directory's
``conftest.py``.
"""

from __future__ import annotations

import struct

import torch

from coro.core.models import SpeakerSegment

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8
CHUNK_AUDIO_SECONDS = CHUNK_LEN * SUBSAMPLING_FACTOR * 0.01


def _make_pcm_bytes(duration_seconds: float) -> bytes:
    n_samples = int(SAMPLE_RATE * duration_seconds)
    return struct.pack(f"<{n_samples}h", *([1000] * n_samples))


# ---------------------------------------------------------------------------
# Test 7: finalize flushes remainder
# ---------------------------------------------------------------------------


def test_finalize_flushes_remainder(diarizer, mock_model, mock_preprocessor, mock_post_processor):
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS / 2)
    diarizer.ingest_pcm_chunk(pcm)
    mock_model.forward_streaming_step.assert_not_called()

    segments = diarizer.finalize()

    mock_model.forward_streaming_step.assert_called_once()
    mock_post_processor.assert_called_once()
    # The post-processor's second segment (1.0-2.0 s) starts past the 0.24 s of audio.
    assert segments == [SpeakerSegment(start=0.0, end=round(CHUNK_AUDIO_SECONDS / 2, 3), speaker=1)]


# ---------------------------------------------------------------------------
# Test 8: finalize returns SpeakerSegments with 1-indexed speakers
# ---------------------------------------------------------------------------


def test_finalize_returns_one_indexed_speaker_segments(
    diarizer, mock_model, mock_preprocessor, mock_post_processor
):
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS)
    diarizer.ingest_pcm_chunk(pcm)

    segments = diarizer.finalize()

    # The post-processor reports speaker index 0; the segment carries speaker 1.
    assert [seg.speaker for seg in segments] == [1]


# ---------------------------------------------------------------------------
# Test 9: finalize with empty buffer returns empty
# ---------------------------------------------------------------------------


def test_finalize_empty_buffer_returns_empty(diarizer, mock_model, mock_post_processor):
    segments = diarizer.finalize()

    mock_model.forward_streaming_step.assert_not_called()
    mock_post_processor.assert_not_called()
    assert segments == []


def test_default_post_process_returns_speaker_segments(mock_model, make_mock_preprocessor):
    """_default_post_process must return SpeakerSegments without relying on the
    post_processor override — exercises ts_vad_post_processing integration."""
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

    mock_preprocessor = make_mock_preprocessor()
    # No post_processor — forces _default_post_process path
    d = StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
    )

    # Seed total_preds with realistic sigmoid-like values: (1, 20, 4)
    torch.manual_seed(0)
    d._total_preds = torch.sigmoid(torch.randn(1, 20, 4))
    d._total_audio_bytes = int(20 * SUBSAMPLING_FACTOR * 0.01 * SAMPLE_RATE * BYTES_PER_SAMPLE)

    segments = d._default_post_process(
        duration=d._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    )

    assert {seg.speaker for seg in segments} == {1, 2, 3, 4}
    assert all(isinstance(seg, SpeakerSegment) for seg in segments)
    assert all(0.0 <= seg.start < seg.end for seg in segments)


def test_default_post_process_uses_resolved_postprocessing_yaml(mock_model, mock_preprocessor):
    """A resolved Diarization Post-Processing Configuration path changes the
    thresholds ts_vad_post_processing actually uses — not just a stored
    attribute. Compares a vendored preset's segments against the raw
    baseline (None) on identical predictions. See ADR 0010."""
    from coro.backends.diarization.nemo.postprocessing import resolve_postprocessing_yaml
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

    torch.manual_seed(0)
    preds = torch.sigmoid(torch.randn(1, 20, 4))
    audio_bytes = int(20 * SUBSAMPLING_FACTOR * 0.01 * SAMPLE_RATE * BYTES_PER_SAMPLE)
    duration = audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    def _segments_with(postprocessing_yaml: str | None) -> list[SpeakerSegment]:
        d = StreamingDiarizer(
            mock_model,
            chunk_len=CHUNK_LEN,
            subsampling_factor=SUBSAMPLING_FACTOR,
            n_spk=4,
            preprocessor=mock_preprocessor,
            postprocessing_yaml=postprocessing_yaml,
        )
        d._total_preds = preds.clone()
        d._total_audio_bytes = audio_bytes
        return d._default_post_process(duration=duration)

    baseline = _segments_with(None)
    preset_path = resolve_postprocessing_yaml("dihard3-dev")
    tuned = _segments_with(preset_path)

    # Both must still be valid SpeakerSegment lists...
    assert all(isinstance(seg, SpeakerSegment) for seg in [*baseline, *tuned])
    # ...but a materially different threshold set (raw 0.5/0.5/no-padding
    # baseline vs. dihard3-dev's padded, hysteretic thresholds) must not
    # silently resolve to the identical segment boundaries on the same
    # random predictions -- otherwise the resolved path was never threaded
    # through to ts_vad_post_processing at all.
    assert [(round(s.start, 6), round(s.end, 6), s.speaker) for s in baseline] != [
        (round(s.start, 6), round(s.end, 6), s.speaker) for s in tuned
    ]
