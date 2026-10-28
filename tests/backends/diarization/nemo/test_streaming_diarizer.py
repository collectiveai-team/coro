"""StreamingDiarizer construction and chunked ingest, with a mocked Sortformer model.

No real NeMo model is loaded; the mocks come from this directory's
``conftest.py``. Finalize and post-processing live in
``test_streaming_diarizer_finalize.py``.
"""

from __future__ import annotations

import struct

import torch


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8
CHUNK_RIGHT_CONTEXT = 1
CHUNK_AUDIO_SECONDS = CHUNK_LEN * SUBSAMPLING_FACTOR * 0.01
CHUNK_AUDIO_BYTES = int(CHUNK_AUDIO_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
RIGHT_CONTEXT_SECONDS = CHUNK_RIGHT_CONTEXT * SUBSAMPLING_FACTOR * 0.01
RIGHT_CONTEXT_BYTES = int(RIGHT_CONTEXT_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
MIN_CHUNK_BYTES = CHUNK_AUDIO_BYTES + RIGHT_CONTEXT_BYTES
N_MEL_FEATURES = 128


def _make_pcm_bytes_n(n_bytes: int) -> bytes:
    n_samples = n_bytes // BYTES_PER_SAMPLE
    return struct.pack(f"<{n_samples}h", *([1000] * n_samples))


# ---------------------------------------------------------------------------
# Test 1: Constructor initialises streaming state
# ---------------------------------------------------------------------------


def test_constructor_initializes_streaming_state(
    mock_model,
    mock_preprocessor,
    mock_post_processor,
):
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

    StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )

    mock_model.sortformer_modules.init_streaming_state.assert_called_once_with(
        batch_size=1,
        async_streaming=False,
        device=mock_model.device,
    )


# ---------------------------------------------------------------------------
# Test 2: ingest processes one chunk
# ---------------------------------------------------------------------------


def test_ingest_processes_one_chunk(diarizer, mock_model, mock_preprocessor):
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES)
    diarizer.ingest_pcm_chunk(pcm)

    mock_preprocessor.assert_called_once()
    mock_model.forward_streaming_step.assert_called_once()
    assert len(diarizer._pcm_buffer) == RIGHT_CONTEXT_BYTES


# ---------------------------------------------------------------------------
# Test 3: ingest buffers until chunk ready
# ---------------------------------------------------------------------------


def test_ingest_buffers_until_chunk_ready(diarizer, mock_model, mock_preprocessor):
    part_pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES // 2)

    diarizer.ingest_pcm_chunk(part_pcm)
    mock_model.forward_streaming_step.assert_not_called()

    diarizer.ingest_pcm_chunk(part_pcm)
    mock_model.forward_streaming_step.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: ingest processes multiple chunks in one call
# ---------------------------------------------------------------------------


def test_ingest_processes_multiple_chunks(diarizer, mock_model, mock_preprocessor):
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES * 2)
    diarizer.ingest_pcm_chunk(pcm)

    assert mock_model.forward_streaming_step.call_count == 2
    assert len(diarizer._pcm_buffer) == RIGHT_CONTEXT_BYTES * 2


def test_forward_receives_empty_total_preds_each_chunk(diarizer, mock_model):
    """Only the current prediction chunk should live on the model device.

    NeMo only uses total_preds for concatenation after inference, so passing an
    empty tensor every step prevents cumulative GPU prediction history.
    """
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES * 2)
    diarizer.ingest_pcm_chunk(pcm)

    first_total_preds = mock_model.forward_streaming_step.call_args_list[0][0][3]
    second_total_preds = mock_model.forward_streaming_step.call_args_list[1][0][3]

    assert first_total_preds.shape == (1, 0, 4)
    assert second_total_preds.shape == (1, 0, 4)
    assert len(diarizer._pred_chunks) == 2
    assert diarizer._pred_chunks[0].device.type == "cpu"
    assert diarizer._pred_chunks[1].device.type == "cpu"


def test_chunk_processing_runs_with_grad_disabled(diarizer, mock_model, mock_preprocessor):
    """Eval mode alone does not disable autograd memory retention."""
    diarizer.ingest_pcm_chunk(_make_pcm_bytes_n(MIN_CHUNK_BYTES))

    assert mock_preprocessor.grad_enabled == [False]
    assert mock_model.grad_enabled_during_forward == [False]


def test_preprocessor_receives_normalized_float_audio(diarizer, mock_preprocessor):
    diarizer.ingest_pcm_chunk(_make_pcm_bytes_n(MIN_CHUNK_BYTES))

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
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES)
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
    """Each call to forward_streaming_step receives mel from the PCM passed
    to _process_chunk — no manual stacking.  Every chunk includes right-context
    bytes (overlap with the next chunk) giving them all the same mel frame count."""
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES * 2)
    diarizer.ingest_pcm_chunk(pcm)

    assert mock_preprocessor.call_count == 2

    first_mel, _ = mock_preprocessor.outputs[0]
    second_mel, _ = mock_preprocessor.outputs[1]

    first_signal = mock_model.forward_streaming_step.call_args_list[0][0][0]
    second_signal = mock_model.forward_streaming_step.call_args_list[1][0][0]

    # Each call's time dimension matches only its own mel output — no manual stacking
    assert first_signal.shape[1] == first_mel.shape[2]
    assert second_signal.shape[1] == second_mel.shape[2]
    # Both chunks include right-context overlap so they have same frame count
    assert first_signal.shape[1] == second_signal.shape[1]


# ---------------------------------------------------------------------------
# Test 10: two instances do not share state
# ---------------------------------------------------------------------------


def test_two_instances_independent(
    mock_model,
    mock_preprocessor,
    mock_post_processor,
    make_mock_model,
    make_mock_preprocessor,
    make_mock_post_processor,
):
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

    mock_model2 = make_mock_model()

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
        preprocessor=make_mock_preprocessor(),
        post_processor=make_mock_post_processor(),
    )

    d1.ingest_pcm_chunk(_make_pcm_bytes_n(MIN_CHUNK_BYTES))
    d2.ingest_pcm_chunk(_make_pcm_bytes_n(MIN_CHUNK_BYTES // 2))

    mock_model.forward_streaming_step.assert_called_once()
    mock_model2.forward_streaming_step.assert_not_called()
    assert len(d1._pcm_buffer) == RIGHT_CONTEXT_BYTES
    assert len(d2._pcm_buffer) > 0


# ---------------------------------------------------------------------------
# Regression tests for warmup bugs fixed in production
# ---------------------------------------------------------------------------


def test_preprocessor_called_with_kwargs_only(diarizer, mock_model, mock_preprocessor):
    """NeMo's @typecheck decorator rejects positional args — preprocessor must be called
    with input_signal= and length= as keyword arguments."""
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES)
    diarizer.ingest_pcm_chunk(pcm)

    call_kwargs = mock_preprocessor.call_args.kwargs
    assert "input_signal" in call_kwargs, "input_signal must be passed as kwarg"
    assert "length" in call_kwargs, "length must be passed as kwarg"
    assert mock_preprocessor.call_args.args == (), "no positional args allowed"


def test_total_preds_never_none_on_construction(mock_model, mock_preprocessor, mock_post_processor):
    """total_preds must be a (1, 0, n_spk) zero tensor at construction time,
    not None — torch.cat rejects None elements."""
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

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
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

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
    d.ingest_pcm_chunk(_make_pcm_bytes_n(MIN_CHUNK_BYTES))

    call_kwargs = mock_preprocessor.call_args.kwargs
    assert call_kwargs["length"].device.type == device.type


# ---------------------------------------------------------------------------
# Regression: right-context offset passed to forward_streaming_step
# ---------------------------------------------------------------------------


def test_right_context_offset_in_ingest_chunks(diarizer, mock_model, mock_post_processor):
    """Every chunk processed in ingest_pcm_chunk includes right-context overlap,
    so right_offset > 0.  Only the finalize flush uses right_offset=0."""
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES * 2)
    diarizer.ingest_pcm_chunk(pcm)
    assert mock_model.forward_streaming_step.call_count == 2

    first_kwargs = mock_model.forward_streaming_step.call_args_list[0][1]
    second_kwargs = mock_model.forward_streaming_step.call_args_list[1][1]

    # Both ingest chunks include right context
    assert first_kwargs.get("right_offset", 0) > 0
    assert second_kwargs.get("right_offset", 0) > 0

    # Finalize flushes remainder with right_offset=0
    diarizer.finalize()
    finalize_kwargs = mock_model.forward_streaming_step.call_args_list[2][1]
    assert finalize_kwargs.get("right_offset", 0) == 0


def test_right_context_value_matches_config(diarizer, mock_model):
    """right_offset matches chunk_right_context * subsampling_factor."""
    expected_offset = CHUNK_RIGHT_CONTEXT * SUBSAMPLING_FACTOR

    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES)
    diarizer.ingest_pcm_chunk(pcm)

    call_kwargs = mock_model.forward_streaming_step.call_args_list[0][1]
    assert call_kwargs["right_offset"] == expected_offset


def test_total_audio_bytes_excludes_right_context_overlap(diarizer, mock_model):
    """_total_audio_bytes counts unique PCM per chunk; right-context bytes remain
    in the buffer (uncounted) for overlap into the next chunk."""
    pcm = _make_pcm_bytes_n(MIN_CHUNK_BYTES * 2)
    total_input = len(pcm)

    diarizer.ingest_pcm_chunk(pcm)

    # Each chunk contributes _chunk_audio_bytes to the total; right-context
    # overlap stays in the buffer as the left part of the next chunk.
    n_chunks = mock_model.forward_streaming_step.call_count
    expected_unique = n_chunks * CHUNK_AUDIO_BYTES
    assert diarizer._total_audio_bytes == expected_unique
    # Buffer holds the uncounted right-context remainder
    assert len(diarizer._pcm_buffer) == total_input - expected_unique
