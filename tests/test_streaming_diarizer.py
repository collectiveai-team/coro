"""Tests for StreamingDiarizer with mocked Sortformer model.

No real NeMo model is loaded. All model interactions use mock objects.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import pytest
import torch

from asr_diar_server.core.types import SpeakerSegment

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8
CHUNK_AUDIO_SECONDS = CHUNK_LEN * SUBSAMPLING_FACTOR * 0.01
CHUNK_AUDIO_BYTES = int(CHUNK_AUDIO_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
LEFT_CONTEXT_FRAMES = 99
N_MEL_FEATURES = 128


def _make_mock_model():
    model = MagicMock()
    model.device = torch.device("cpu")

    sortformer_modules = MagicMock()
    sortformer_modules.chunk_len = CHUNK_LEN
    sortformer_modules.subsampling_factor = SUBSAMPLING_FACTOR
    sortformer_modules.n_spk = 4
    sortformer_modules.fc_d_model = 512
    initial_state = {"step": 0}
    sortformer_modules.init_streaming_state.return_value = initial_state
    model.sortformer_modules = sortformer_modules

    step_count = [0]

    def _forward_streaming_step(
        processed_signal, processed_signal_length, streaming_state, total_preds, **kwargs
    ):
        step_count[0] += 1
        new_state = {"step": step_count[0]}
        if total_preds is None:
            total_preds = torch.zeros(1, 4, 100)
        total_preds = total_preds + torch.rand_like(total_preds) * 0.01
        return new_state, total_preds

    model.forward_streaming_step = MagicMock(side_effect=_forward_streaming_step)
    model._step_count = step_count
    return model


def _make_mock_preprocessor():
    outputs = []

    def _process(audio_signal, length):
        n_samples = audio_signal.shape[-1]
        n_mel_frames = n_samples // 160
        if n_mel_frames == 0:
            n_mel_frames = 1
        mel = torch.randn(1, N_MEL_FEATURES, n_mel_frames)
        mel_len = torch.tensor([n_mel_frames])
        outputs.append((mel.clone(), mel_len.clone()))
        return mel, mel_len

    preprocessor = MagicMock(side_effect=_process)
    preprocessor.outputs = outputs
    return preprocessor


def _make_mock_post_processor():
    def _post_process(total_preds, n_spk):
        return [
            (0.0, 1.0, 0),
            (1.0, 2.0, 1),
        ]

    return MagicMock(side_effect=_post_process)


def _make_pcm_bytes(duration_seconds: float) -> bytes:
    n_samples = int(SAMPLE_RATE * duration_seconds)
    return struct.pack(f"<{n_samples}h", *([1000] * n_samples))


@pytest.fixture()
def mock_model():
    return _make_mock_model()


@pytest.fixture()
def mock_preprocessor():
    return _make_mock_preprocessor()


@pytest.fixture()
def mock_post_processor():
    return _make_mock_post_processor()


@pytest.fixture()
def diarizer(mock_model, mock_preprocessor, mock_post_processor):
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizer

    return StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )


# ---------------------------------------------------------------------------
# Test 1: Constructor initialises streaming state
# ---------------------------------------------------------------------------


def test_constructor_initializes_streaming_state(
    mock_model, mock_preprocessor, mock_post_processor,
):
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizer

    StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )

    mock_model.sortformer_modules.init_streaming_state.assert_called_once_with(
        batch_size=1, async_streaming=False, device=mock_model.device,
    )


# ---------------------------------------------------------------------------
# Test 2: ingest processes one chunk
# ---------------------------------------------------------------------------


def test_ingest_processes_one_chunk(diarizer, mock_model, mock_preprocessor):
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS)
    diarizer.ingest_pcm_chunk(pcm)

    mock_preprocessor.assert_called_once()
    mock_model.forward_streaming_step.assert_called_once()
    assert diarizer._pcm_buffer == b""


# ---------------------------------------------------------------------------
# Test 3: ingest buffers until chunk ready
# ---------------------------------------------------------------------------


def test_ingest_buffers_until_chunk_ready(diarizer, mock_model, mock_preprocessor):
    half_pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS / 2)

    diarizer.ingest_pcm_chunk(half_pcm)
    mock_model.forward_streaming_step.assert_not_called()

    diarizer.ingest_pcm_chunk(half_pcm)
    mock_model.forward_streaming_step.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: ingest processes multiple chunks in one call
# ---------------------------------------------------------------------------


def test_ingest_processes_multiple_chunks(diarizer, mock_model, mock_preprocessor):
    two_chunks = _make_pcm_bytes(CHUNK_AUDIO_SECONDS * 2)
    diarizer.ingest_pcm_chunk(two_chunks)

    assert mock_model.forward_streaming_step.call_count == 2
    assert diarizer._pcm_buffer == b""


# ---------------------------------------------------------------------------
# Test 5: left context zero-padded for first chunk
# ---------------------------------------------------------------------------


def test_left_context_zero_padded_for_first_chunk(diarizer, mock_model, mock_preprocessor):
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS)
    diarizer.ingest_pcm_chunk(pcm)

    first_call = mock_model.forward_streaming_step.call_args
    signal = first_call[0][0]

    left_ctx = signal[:, :, :LEFT_CONTEXT_FRAMES]
    assert torch.all(left_ctx == 0)


# ---------------------------------------------------------------------------
# Test 6: left context carried between chunks
# ---------------------------------------------------------------------------


def test_left_context_carried_between_chunks(diarizer, mock_model, mock_preprocessor):
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS * 2)
    diarizer.ingest_pcm_chunk(pcm)

    assert mock_preprocessor.call_count == 2
    first_mel, _ = mock_preprocessor.outputs[0]

    second_call = mock_model.forward_streaming_step.call_args_list[1]
    second_signal = second_call[0][0]

    carried_frames = min(LEFT_CONTEXT_FRAMES, first_mel.shape[-1])
    expected_left_ctx = first_mel[:, :, -carried_frames:]
    actual_left_ctx = second_signal[:, :, :carried_frames]
    assert torch.allclose(expected_left_ctx, actual_left_ctx)


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
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, SpeakerSegment)


# ---------------------------------------------------------------------------
# Test 8: finalize returns SpeakerSegments with 1-indexed speakers
# ---------------------------------------------------------------------------


def test_finalize_returns_one_indexed_speaker_segments(
    diarizer, mock_model, mock_preprocessor, mock_post_processor
):
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS)
    diarizer.ingest_pcm_chunk(pcm)

    segments = diarizer.finalize()

    assert len(segments) > 0
    for seg in segments:
        assert seg.speaker >= 1


# ---------------------------------------------------------------------------
# Test 9: finalize with empty buffer returns empty
# ---------------------------------------------------------------------------


def test_finalize_empty_buffer_returns_empty(diarizer, mock_model, mock_post_processor):
    segments = diarizer.finalize()

    mock_model.forward_streaming_step.assert_not_called()
    mock_post_processor.assert_not_called()
    assert segments == []


# ---------------------------------------------------------------------------
# Test 10: two instances do not share state
# ---------------------------------------------------------------------------


def test_two_instances_independent(mock_model, mock_preprocessor, mock_post_processor):
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizer

    mock_model2 = _make_mock_model()

    d1 = StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )
    d2 = StreamingDiarizer(
        mock_model2,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=_make_mock_preprocessor(),
        post_processor=_make_mock_post_processor(),
    )

    d1.ingest_pcm_chunk(_make_pcm_bytes(CHUNK_AUDIO_SECONDS))
    d2.ingest_pcm_chunk(_make_pcm_bytes(CHUNK_AUDIO_SECONDS / 2))

    mock_model.forward_streaming_step.assert_called_once()
    mock_model2.forward_streaming_step.assert_not_called()
    assert d1._pcm_buffer == b""
    assert len(d2._pcm_buffer) > 0
