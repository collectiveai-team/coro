"""Mocked Sortformer model, preprocessor and post-processor for StreamingDiarizer tests.

No real NeMo model is loaded. ``CHUNK_LEN``, ``SUBSAMPLING_FACTOR`` and
``N_MEL_FEATURES`` match the constants of the test modules that use these
fixtures.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

CHUNK_LEN = 6
SUBSAMPLING_FACTOR = 8
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


@pytest.fixture()
def make_mock_model():
    return _make_mock_model


@pytest.fixture()
def make_mock_preprocessor():
    return _make_mock_preprocessor


@pytest.fixture()
def make_mock_post_processor():
    return _make_mock_post_processor


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
    from coro.backends.diarization.nemo.streaming import StreamingDiarizer

    return StreamingDiarizer(
        mock_model,
        chunk_len=CHUNK_LEN,
        subsampling_factor=SUBSAMPLING_FACTOR,
        n_spk=4,
        preprocessor=mock_preprocessor,
        post_processor=mock_post_processor,
    )
