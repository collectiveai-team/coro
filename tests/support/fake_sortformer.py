"""In-memory Sortformer streaming model, mel preprocessor and post-processor.

``FakeStreamingSortformer`` implements ``StreamingSortformerModel`` and
``FakeMelPreprocessor`` implements ``MelPreprocessor``, both from
``coro.backends.diarization.nemo.sortformer``. They run no NeMo code and
return deterministic tensors. Each raises ``ContractError`` on an input that
NeMo rejects or that wastes memory on a real model, so a diarizer that breaks
the contract fails the test that drives it:

- the preprocessor takes keyword arguments only (NeMo's ``@typecheck``), a
  float32 signal in [-1, 1], and a length tensor on the signal's device;
- the model takes ``(batch, time, features)`` features whose length matches
  the signal, the streaming state its previous step returned, an empty
  ``(1, 0, n_spk)`` ``total_preds`` (a fuller one keeps prediction history on
  the device), and runs with autograd off;
- ``init_streaming_state`` is asked for one synchronous state on the model's
  device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

N_MEL_FEATURES = 128
MEL_HOP_SAMPLES = 160


class ContractError(AssertionError):
    """The diarizer called a fake in a way the real NeMo object would not accept."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


@dataclass(frozen=True)
class FakeStreamingState:
    """The speaker-cache state after ``steps`` streaming steps."""

    steps: int


class FakeSortformerModules:
    """``model.sortformer_modules``: the streaming parameters and the state factory."""

    def __init__(self, *, device: torch.device, subsampling_factor: int, n_spk: int) -> None:
        self.subsampling_factor = subsampling_factor
        self.n_spk = n_spk
        self._device = device

    def _check_streaming_parameters(self) -> None:
        return None

    def init_streaming_state(
        self, *, batch_size: int, async_streaming: bool, device: torch.device
    ) -> FakeStreamingState:
        """Return the state before any step."""
        _require(batch_size == 1, f"batch_size={batch_size}, expected 1")
        _require(not async_streaming, "async streaming state requested")
        _require(device == self._device, f"state on {device}, model on {self._device}")
        return FakeStreamingState(steps=0)


class FakeStreamingSortformer:
    """A streaming Sortformer that emits one prediction frame per ``subsampling_factor`` mel frames.

    The model drops the ``right_offset`` right-context frames before
    predicting, as NeMo does. Without ``predictions`` every frame is a
    near-silent 0.01 for each speaker. With ``predictions``
    (``(1, frames, n_spk)``) the frames are handed out in order, so the
    diarizer's output is fixed whatever its chunking. One fake serves one
    diarizer, since it checks the state against the state it last returned.
    """

    def __init__(
        self,
        predictions: torch.Tensor | None = None,
        *,
        subsampling_factor: int = 8,
        n_spk: int = 4,
    ) -> None:
        self.device = torch.device("cpu")
        self.sortformer_modules = FakeSortformerModules(
            device=self.device, subsampling_factor=subsampling_factor, n_spk=n_spk
        )
        self._predictions = predictions
        self._handed_out = 0
        self._state = FakeStreamingState(steps=0)

    def forward_streaming_step(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
        streaming_state: object,
        total_preds: torch.Tensor,
        /,
        *,
        left_offset: int,
        right_offset: int,
    ) -> tuple[object, torch.Tensor]:
        """Append this chunk's predictions to ``total_preds``, as NeMo does."""
        n_spk = self.sortformer_modules.n_spk
        _require(not torch.is_grad_enabled(), "streaming step run with autograd on")
        _require(
            processed_signal.ndim == 3 and processed_signal.shape[2] == N_MEL_FEATURES,
            f"features {tuple(processed_signal.shape)} are not (batch, time, {N_MEL_FEATURES})",
        )
        _require(
            int(processed_signal_length.item()) == processed_signal.shape[1],
            "feature length does not match the features",
        )
        _require(streaming_state == self._state, f"stale state {streaming_state}")
        _require(
            tuple(total_preds.shape) == (1, 0, n_spk),
            f"total_preds {tuple(total_preds.shape)} keeps history on the device",
        )
        frames = (
            processed_signal.shape[1] - right_offset
        ) // self.sortformer_modules.subsampling_factor
        self._state = FakeStreamingState(steps=self._state.steps + 1)
        return self._state, torch.cat([total_preds, self._next_frames(frames)], dim=1)

    def _next_frames(self, frames: int) -> torch.Tensor:
        if self._predictions is None:
            return torch.full((1, frames, self.sortformer_modules.n_spk), 0.01)
        chunk = self._predictions[:, self._handed_out : self._handed_out + frames]
        self._handed_out += frames
        return chunk


class FakeMelPreprocessor:
    """Silent log-mel features: one frame per 160 samples, ``(batch, features, time)``."""

    def __call__(
        self, *, input_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero features for ``input_signal``."""
        _require(not torch.is_grad_enabled(), "preprocessor run with autograd on")
        _require(input_signal.dtype == torch.float32, f"signal is {input_signal.dtype}")
        _require(float(input_signal.abs().max()) <= 1.0, "signal is not normalised to [-1, 1]")
        _require(length.device == input_signal.device, "length and signal on different devices")
        frames = max(1, input_signal.shape[-1] // MEL_HOP_SAMPLES)
        return torch.zeros(1, N_MEL_FEATURES, frames), torch.tensor([frames])


@dataclass
class FixedPostProcessor:
    """A ``PostProcessor`` that returns the same native segments for any predictions."""

    segments: list[tuple[float, float, int]]
    received: list[torch.Tensor] = field(default_factory=list)

    def __call__(self, total_preds: torch.Tensor, n_spk: int) -> list[tuple[float, float, int]]:
        """Return ``segments`` and keep the predictions it was given."""
        self.received.append(total_preds)
        return list(self.segments)
