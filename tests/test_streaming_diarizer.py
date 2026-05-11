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
N_MEL_FEATURES = 128


def _make_mock_model():
    model = MagicMock()
    model.device = torch.device("cpu")
    model.grad_enabled_during_forward = []

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
        model.grad_enabled_during_forward.append(torch.is_grad_enabled())
        # processed_signal arrives as (batch, time, features) after transpose in _process_chunk.
        # total_preds is now initialised as (1, 0, n_spk) — never None.
        step_count[0] += 1
        new_state = {"step": step_count[0]}
        chunk_frames = 4  # arbitrary small chunk pred length
        chunk_preds = torch.rand(1, chunk_frames, 4) * 0.01
        new_total_preds = torch.cat([total_preds, chunk_preds], dim=1)
        return new_state, new_total_preds

    model.forward_streaming_step = MagicMock(side_effect=_forward_streaming_step)
    model._step_count = step_count
    return model


def _make_mock_preprocessor():
    outputs = []
    grad_enabled = []
    inputs = []

    def _process(*, input_signal, length):
        grad_enabled.append(torch.is_grad_enabled())
        inputs.append(input_signal.clone())
        # Called with kwargs — matches the NeMo typecheck requirement.
        n_samples = input_signal.shape[-1]
        n_mel_frames = n_samples // 160
        if n_mel_frames == 0:
            n_mel_frames = 1
        mel = torch.randn(1, N_MEL_FEATURES, n_mel_frames)
        mel_len = torch.tensor([n_mel_frames])
        outputs.append((mel.clone(), mel_len.clone()))
        return mel, mel_len

    preprocessor = MagicMock(side_effect=_process)
    preprocessor.outputs = outputs
    preprocessor.grad_enabled = grad_enabled
    preprocessor.inputs = inputs
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


def test_forward_receives_empty_total_preds_each_chunk(diarizer, mock_model):
    """Only the current prediction chunk should live on the model device.

    NeMo only uses total_preds for concatenation after inference, so passing an
    empty tensor every step prevents cumulative GPU prediction history.
    """
    two_chunks = _make_pcm_bytes(CHUNK_AUDIO_SECONDS * 2)
    diarizer.ingest_pcm_chunk(two_chunks)

    first_total_preds = mock_model.forward_streaming_step.call_args_list[0][0][3]
    second_total_preds = mock_model.forward_streaming_step.call_args_list[1][0][3]

    assert first_total_preds.shape == (1, 0, 4)
    assert second_total_preds.shape == (1, 0, 4)
    assert len(diarizer._pred_chunks) == 2
    assert diarizer._pred_chunks[0].device.type == "cpu"
    assert diarizer._pred_chunks[1].device.type == "cpu"


def test_chunk_processing_runs_with_grad_disabled(diarizer, mock_model, mock_preprocessor):
    """Eval mode alone does not disable autograd memory retention."""
    diarizer.ingest_pcm_chunk(_make_pcm_bytes(CHUNK_AUDIO_SECONDS))

    assert mock_preprocessor.grad_enabled == [False]
    assert mock_model.grad_enabled_during_forward == [False]


def test_preprocessor_receives_normalized_float_audio(diarizer, mock_preprocessor):
    diarizer.ingest_pcm_chunk(_make_pcm_bytes(CHUNK_AUDIO_SECONDS))

    audio_signal = mock_preprocessor.inputs[0]
    assert audio_signal.dtype == torch.float32
    assert torch.max(torch.abs(audio_signal)).item() <= 1.0


# ---------------------------------------------------------------------------
# Test 5: signal passed to forward_streaming_step is time-first (batch, time, features)
# ---------------------------------------------------------------------------


def test_signal_is_time_first_on_first_chunk(diarizer, mock_model, mock_preprocessor):
    """forward_streaming_step expects (batch, time, features).
    The preprocessor returns (batch, features, time) — we must transpose before passing.
    No external left-context is prepended; streaming_state carries history internally.
    """
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS)
    diarizer.ingest_pcm_chunk(pcm)

    first_call = mock_model.forward_streaming_step.call_args
    signal = first_call[0][0]

    # (batch, time, features): dim 2 must be N_MEL_FEATURES
    assert signal.ndim == 3
    assert signal.shape[2] == N_MEL_FEATURES
    # No left-context prepended — time frames equal what preprocessor produced
    mel, _ = mock_preprocessor.outputs[0]
    assert signal.shape[1] == mel.shape[2]


# ---------------------------------------------------------------------------
# Test 6: each chunk passes only its own frames — no accumulation across chunks
# ---------------------------------------------------------------------------


def test_each_chunk_passes_only_its_own_frames(diarizer, mock_model, mock_preprocessor):
    """streaming_state carries left-context history internally; each call to
    forward_streaming_step receives only the current chunk's mel frames."""
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS * 2)
    diarizer.ingest_pcm_chunk(pcm)

    assert mock_preprocessor.call_count == 2

    first_mel, _ = mock_preprocessor.outputs[0]
    second_mel, _ = mock_preprocessor.outputs[1]

    first_signal = mock_model.forward_streaming_step.call_args_list[0][0][0]
    second_signal = mock_model.forward_streaming_step.call_args_list[1][0][0]

    # Each call's time dimension matches only its own mel output — no growth
    assert first_signal.shape[1] == first_mel.shape[2]
    assert second_signal.shape[1] == second_mel.shape[2]


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


# ---------------------------------------------------------------------------
# Regression tests for warmup bugs fixed in production
# ---------------------------------------------------------------------------


def test_preprocessor_called_with_kwargs_only(diarizer, mock_model, mock_preprocessor):
    """NeMo's @typecheck decorator rejects positional args — preprocessor must be called
    with input_signal= and length= as keyword arguments."""
    pcm = _make_pcm_bytes(CHUNK_AUDIO_SECONDS)
    diarizer.ingest_pcm_chunk(pcm)

    call_kwargs = mock_preprocessor.call_args.kwargs
    assert "input_signal" in call_kwargs, "input_signal must be passed as kwarg"
    assert "length" in call_kwargs, "length must be passed as kwarg"
    assert mock_preprocessor.call_args.args == (), "no positional args allowed"


def test_total_preds_never_none_on_construction(mock_model, mock_preprocessor, mock_post_processor):
    """total_preds must be a (1, 0, n_spk) zero tensor at construction time,
    not None — torch.cat rejects None elements."""
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizer

    d = StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )

    assert d._total_preds is not None
    assert isinstance(d._total_preds, torch.Tensor)
    assert d._total_preds.shape == (1, 0, 4)



def test_length_tensor_on_same_device_as_audio(mock_model, mock_preprocessor, mock_post_processor):
    """The length tensor passed to the preprocessor must be on the model's device
    to avoid cross-device matmul errors."""
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizer

    device = torch.device("cpu")
    mock_model.device = device

    d = StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )
    d.ingest_pcm_chunk(_make_pcm_bytes(CHUNK_AUDIO_SECONDS))

    call_kwargs = mock_preprocessor.call_args.kwargs
    assert call_kwargs["length"].device.type == device.type


def test_default_post_process_returns_speaker_segments(mock_model):
    """_default_post_process must return SpeakerSegments without relying on the
    post_processor override — exercises ts_vad_post_processing integration."""
    from asr_diar_server.backends.nemo_streaming import StreamingDiarizer

    mock_preprocessor = _make_mock_preprocessor()
    # No post_processor — forces _default_post_process path
    d = StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
    )

    # Seed total_preds with realistic sigmoid-like values: (1, 20, 4)
    d._total_preds = torch.sigmoid(torch.randn(1, 20, 4))
    d._total_audio_bytes = int(20 * SUBSAMPLING_FACTOR * 0.01 * SAMPLE_RATE * BYTES_PER_SAMPLE)

    segments = d._default_post_process(
        duration=d._total_audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    )

    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, SpeakerSegment)
        assert seg.speaker >= 1
        assert seg.start >= 0.0
        assert seg.end > seg.start
