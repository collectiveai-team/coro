"""Prompt conditioning for the onnx-parakeet-prompt backend.

Holds the checkpoint's ``model.prompt_kernel`` MLP replicated in NumPy, the
artifacts it loads from, and the ``NemoConformerRnnt`` subclass that applies
it to the encoder output. See ``coro/backends/asr/onnx_parakeet_prompt.py``'s
module docstring for the artifact directory contract and the design.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, override

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt
    from onnx_asr.asr import Preprocessor
    from onnx_asr.onnx import OnnxSessionOptions

PROMPT_KERNEL_WEIGHTS_FILENAME = "prompt_kernel_cache.npz"
PROMPT_KERNEL_METADATA_FILENAME = "prompt_kernel_cache.json"

# prompt_kernel.state_dict() key names for a two-layer nn.Sequential
# (Linear -> ReLU -> Linear, NeMo's own indexing), verified against the real
# checkpoint (see findings.md).
_PROMPT_KERNEL_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")


class PromptSlot(Protocol):
    """An ASR instance carrying the call-scoped prompt id its ``_encode`` reads."""

    _prompt_id: int | None


def apply_prompt_kernel(
    encoder_out: np.ndarray, prompt_id: int, weights: dict[str, np.ndarray]
) -> np.ndarray:
    """Force a target language by running the checkpoint's prompt_kernel MLP.

    Concatenates a per-timestep one-hot language vector onto the encoder's
    hidden state, then applies ``Linear -> ReLU -> Linear`` (the checkpoint's
    ``model.prompt_kernel``, replicated in NumPy from its ``state_dict()``
    rather than exported as a second ONNX graph -- see the module docstring).
    findings.md verified this substantially changes the encoder representation
    (mean abs diff ~0.05 vs. the signal's own ~0.18 magnitude) and is
    deterministic (same language twice -> zero diff).

    Args:
        encoder_out: Encoder output, shape ``(batch, time, hidden)``.
        prompt_id: One-hot index into the checkpoint's prompt dictionary.
        weights: ``prompt_kernel.npz`` arrays (``0.weight``, ``0.bias``,
            ``2.weight``, ``2.bias``; PyTorch ``nn.Linear`` layout, i.e.
            ``(out_features, in_features)``).

    Returns:
        The prompt-conditioned hidden state, same shape as ``encoder_out``.

    """
    _batch, _time_steps, hidden = encoder_out.shape
    w0, b0 = weights["0.weight"], weights["0.bias"]
    w2, b2 = weights["2.weight"], weights["2.bias"]
    num_prompts = w0.shape[1] - hidden

    one_hot = np.zeros((*encoder_out.shape[:2], num_prompts), dtype=np.float32)
    one_hot[:, :, prompt_id] = 1.0
    hidden_state = np.concatenate([encoder_out, one_hot], axis=-1)
    hidden_state = np.maximum(hidden_state @ w0.T + b0, 0.0)
    hidden_state = hidden_state @ w2.T + b2
    return hidden_state.astype(np.float32)


class ForcedPrompt:
    """Sets/clears a prompt-conditioned ASR instance's ``_prompt_id`` around one call.

    ``_prompt_id`` is call-scoped mutable instance state rather than a
    ``_encode`` argument: ``onnx_asr``'s ``recognize_batch`` calls ``_encode``
    before any kwargs reach ``_decoding``, so there is no fixed-signature way
    to thread it through. This is exactly why
    :class:`~coro.backends.asr.onnx_parakeet_prompt.OnnxParakeetPromptASRAdapter`
    declares a serialised Adapter Concurrency Policy -- two overlapping calls
    would race on this attribute.
    """

    def __init__(self, asr: PromptSlot, prompt_id: int) -> None:
        self._asr = asr
        self._prompt_id = prompt_id

    def __enter__(self) -> None:
        self._asr._prompt_id = self._prompt_id

    def __exit__(self, *exc_info: object) -> None:
        self._asr._prompt_id = None


def prompt_conditioned_asr_class() -> type:
    """Build the ``NemoConformerRnnt`` subclass that forces a prompt language.

    Defined inside a function (rather than at module scope) so importing this
    module does not require ``onnx_asr``/``onnxruntime`` to be installed
    unless the backend is actually selected -- every ``coro.backends.asr``
    module defers its runtime import the same way (see ``factory.py``'s
    module docstring).
    """
    from onnx_asr.models.nemo import NemoConformerRnnt

    class _NemoConformerRnntWithPrompt(NemoConformerRnnt):
        """``NemoConformerRnnt`` whose encoder output is prompt-conditioned.

        Only ``_encode`` is overridden -- the inherited RNNT greedy-decode
        loop, preprocessing and vocabulary handling are unmodified (see the
        module docstring).
        """

        def __init__(
            self,
            model_files: dict[str, Path],
            preprocessor_factory: Callable[[str], Preprocessor],
            onnx_options: OnnxSessionOptions,
            *,
            prompt_kernel_weights: dict[str, np.ndarray],
        ) -> None:
            super().__init__(model_files, preprocessor_factory, onnx_options)
            self._prompt_weights = prompt_kernel_weights
            self._prompt_id: int | None = None

        def forced_prompt(self, prompt_id: int) -> ForcedPrompt:
            """Context manager forcing ``prompt_id`` for one ``recognize_batch`` call."""
            return ForcedPrompt(self, prompt_id)

        @override
        def _encode(
            self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
        ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
            encoder_out, encoder_out_lens = super()._encode(features, features_lens)
            if self._prompt_id is None:
                msg = (
                    "_NemoConformerRnntWithPrompt._encode called with no forced "
                    "language in scope; call within `asr.forced_prompt(prompt_id)`."
                )
                raise RuntimeError(msg)
            return (
                apply_prompt_kernel(encoder_out, self._prompt_id, self._prompt_weights),
                encoder_out_lens,
            )

    return _NemoConformerRnntWithPrompt


def load_prompt_kernel(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load the prompt_kernel MLP weights and the checkpoint's prompt dictionary."""
    weights_path = directory / PROMPT_KERNEL_WEIGHTS_FILENAME
    metadata_path = directory / PROMPT_KERNEL_METADATA_FILENAME
    if not weights_path.is_file():
        msg = f"Missing required onnx-parakeet-prompt artifact: {weights_path}"
        raise FileNotFoundError(msg)
    if not metadata_path.is_file():
        msg = f"Missing required onnx-parakeet-prompt artifact: {metadata_path}"
        raise FileNotFoundError(msg)

    with np.load(weights_path) as npz:
        weights = {key: np.asarray(npz[key], dtype=np.float32) for key in _PROMPT_KERNEL_KEYS}
    with metadata_path.open("rt", encoding="utf-8") as f:
        metadata = json.load(f)
    prompt_dictionary = {str(k): int(v) for k, v in metadata["prompt_dictionary"].items()}
    return weights, prompt_dictionary
