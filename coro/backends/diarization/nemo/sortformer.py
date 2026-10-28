"""The slices of a NeMo Sortformer model the diarization adapters call.

``SortformerEncLabelModel`` resolves its submodules through
``torch.nn.Module.__getattr__``, so its static type names none of the members
below; these Protocols name exactly what the batch and streaming Diarization
Flows use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import torch


class SortformerModel(Protocol):
    """The batch Diarization Flow's view of a Sortformer model."""

    @property
    def device(self) -> torch.device:
        """Device holding the model weights."""
        ...

    def forward(
        self, *, audio_signal: torch.Tensor, audio_signal_length: torch.Tensor
    ) -> torch.Tensor:
        """Return ``(batch, frames, speakers)`` speaker-activity sigmoids."""
        ...


class HasSortformerModules(Protocol):
    """Anything carrying a ``sortformer_modules`` submodule to retune."""

    @property
    def sortformer_modules(self) -> object:
        """The submodule carrying the latency-tier streaming parameters."""
        ...


class SortformerStreamingModules(Protocol):
    """``model.sortformer_modules``: owns the streaming parameters and state."""

    def _check_streaming_parameters(self) -> None: ...

    def init_streaming_state(
        self, *, batch_size: int, async_streaming: bool, device: torch.device
    ) -> object:
        """Return a fresh speaker-cache/FIFO streaming state."""
        ...


class StreamingSortformerModel(Protocol):
    """The Streaming Diarization Flow's view of a Sortformer model."""

    @property
    def device(self) -> torch.device:
        """Device holding the model weights."""
        ...

    @property
    def sortformer_modules(self) -> SortformerStreamingModules:
        """The submodule carrying the latency-tier streaming parameters."""
        ...

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
        """Advance the streaming state by one chunk; return it with the chunk's predictions."""
        ...


class SortformerDiarizationModel(SortformerModel, StreamingSortformerModel, Protocol):
    """One loaded Sortformer model, shared by the batch and streaming flows."""


class MelPreprocessor(Protocol):
    """Audio -> log-mel features, as NeMo's ``AudioToMelSpectrogramPreprocessor``."""

    def __call__(
        self, *, input_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features, feature_lengths)`` in ``(batch, features, time)`` layout."""
        ...


# Raw predictions and speaker count -> native diarization segments.
type PostProcessor = Callable[[torch.Tensor, int], Iterable[object]]
