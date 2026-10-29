"""StreamingDiarizer finalize and post-processing, over an in-memory Sortformer.

The fakes come from ``support.fake_sortformer``, partly through this
directory's ``conftest.py``. The fixture's post-processor returns speaker 0
over 0-1 s and speaker 1 over 1-2 s for any predictions. The tests read
the segments ``finalize`` returns and the public ``processed_chunks`` that
the Streaming Pipeline reports.
"""

from __future__ import annotations

import struct

import torch
from support.fake_sortformer import FakeMelPreprocessor, FakeStreamingSortformer

from coro.backends.diarization.nemo.postprocessing import resolve_postprocessing_yaml
from coro.backends.diarization.nemo.streaming import StreamingDiarizer
from coro.core.models import SpeakerSegment

SAMPLE_RATE = 16000
CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8
CHUNK_AUDIO_SECONDS = CHUNK_LEN * SUBSAMPLING_FACTOR * 0.01
PREDICTION_FRAMES = 20
PREDICTIONS_SECONDS = PREDICTION_FRAMES * SUBSAMPLING_FACTOR * 0.01


def pcm_of_seconds(duration_seconds: float) -> bytes:
    n_samples = int(SAMPLE_RATE * duration_seconds)
    return struct.pack(f"<{n_samples}h", *([1000] * n_samples))


def random_activity() -> torch.Tensor:
    """Sigmoid speaker activity, ``(1, 20, 4)``, the same on every call."""
    generator = torch.Generator().manual_seed(0)
    return torch.sigmoid(torch.randn(1, PREDICTION_FRAMES, 4, generator=generator))


def segments_of_default_post_processing(postprocessing_yaml: str | None) -> list[SpeakerSegment]:
    """Stream 1.6 s whose 20 prediction frames are ``random_activity()``, and finalize."""
    diarizer = StreamingDiarizer(
        FakeStreamingSortformer(random_activity()),
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=FakeMelPreprocessor(),
        postprocessing_yaml=postprocessing_yaml,
    )
    diarizer.ingest_pcm_chunk(pcm_of_seconds(PREDICTIONS_SECONDS))
    return diarizer.finalize()


def test_finalize_flushes_remainder(diarizer):
    diarizer.ingest_pcm_chunk(pcm_of_seconds(CHUNK_AUDIO_SECONDS / 2))
    assert diarizer.processed_chunks == 0

    segments = diarizer.finalize()

    assert diarizer.processed_chunks == 1
    # The post-processor's second segment (1.0-2.0 s) starts past the 0.24 s of audio.
    assert segments == [SpeakerSegment(start=0.0, end=round(CHUNK_AUDIO_SECONDS / 2, 3), speaker=1)]


def test_finalize_returns_one_indexed_speaker_segments(diarizer):
    diarizer.ingest_pcm_chunk(pcm_of_seconds(CHUNK_AUDIO_SECONDS))

    segments = diarizer.finalize()

    assert [seg.speaker for seg in segments] == [1]


def test_finalize_empty_buffer_returns_empty(diarizer):
    """The post-processor of the fixture returns two segments for any
    predictions, so an empty result means it never ran."""
    segments = diarizer.finalize()

    assert segments == []
    assert diarizer.processed_chunks == 0


def test_default_post_process_returns_speaker_segments():
    """With no post_processor override, finalize runs ts_vad_post_processing."""
    segments = segments_of_default_post_processing(None)

    assert {seg.speaker for seg in segments} == {1, 2, 3, 4}
    assert all(isinstance(seg, SpeakerSegment) for seg in segments)
    assert all(0.0 <= seg.start < seg.end for seg in segments)


def test_default_post_process_uses_resolved_postprocessing_yaml():
    """A resolved Diarization Post-Processing Configuration changes the
    thresholds ts_vad_post_processing uses, not just a stored attribute.
    See ADR 0010."""
    baseline = segments_of_default_post_processing(None)
    tuned = segments_of_default_post_processing(resolve_postprocessing_yaml("dihard3-dev"))

    assert all(isinstance(seg, SpeakerSegment) for seg in [*baseline, *tuned])
    # The raw 0.5/0.5 no-padding baseline and dihard3-dev's padded, hysteretic
    # thresholds cannot give the same boundaries on these predictions unless
    # the preset never reached ts_vad_post_processing.
    assert [(round(s.start, 6), round(s.end, 6), s.speaker) for s in baseline] != [
        (round(s.start, 6), round(s.end, 6), s.speaker) for s in tuned
    ]
