"""Split cross-attention K/V decode graphs for the onnx-canary-split backend.

Holds the ``NemoConformerAED`` subclass that drives ``xattn_kv.onnx`` and
``decoder_step.onnx`` instead of the fused decoder; see
``coro/backends/asr/onnx_canary_split.py``'s module docstring.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from coro.backends.asr.onnx_canary_split_language import _LANGUAGE_TOKEN_RE

if TYPE_CHECKING:
    from pathlib import Path

# The 16 cross-attention K/V frontier tensors -- see `.tmp/split_canary_decoder.py`'s
# module docstring for why these tensors (not the shallower MatMul outputs) are
# the maximal hoistable cut. Duplicated here (rather than imported) because that
# script lives in `.tmp/`, which this package must not depend on at import time.
_KV_TENSORS: list[str] = []
for _layer in range(8):
    _KV_TENSORS.append(f"/_decoder/layers.{_layer}/second_sub_layer/Transpose_3_output_0")
    _KV_TENSORS.append(f"/_decoder/layers.{_layer}/second_sub_layer/Transpose_2_output_0")


def _split_canary_asr_class() -> type:
    """Build the ``NemoConformerAED`` subclass that drives the split decoder graphs.

    Defined inside a function (rather than at module scope) so importing this
    module does not require ``onnx_asr``/``onnxruntime`` to be installed unless
    the backend is actually selected -- every ``coro.backends.asr`` module
    defers its runtime import the same way (see ``factory.py``'s module
    docstring).
    """
    import onnxruntime as rt
    from onnx_asr.models.nemo import NemoConformerAED
    from onnx_asr.onnx import TensorRtOptions

    class _NemoConformerAEDSplitDecode(NemoConformerAED):
        """``NemoConformerAED`` whose decode step drives split cross-attention K/V graphs.

        ``__init__`` is fully overridden (not extended) to avoid loading the
        fused decoder this class never calls -- see the module docstring.
        Only ``_decode`` is overridden beyond that; ``_encode``, ``_decoding``,
        preprocessing and tokenization are all inherited unmodified.
        """

        def __init__(
            self,
            model_files: dict[str, Path],
            preprocessor_factory: Any,
            onnx_options: Any,
        ) -> None:
            # `_NemoConformer` (NemoConformerAED's other parent) defines no
            # `__init__`, so this reaches `_AsrWithDecoding.__init__` (vocab,
            # config, preprocessor setup) without `NemoConformerAED.__init__`'s
            # unconditional fused-decoder-session build.
            super(NemoConformerAED, self).__init__(model_files, preprocessor_factory, onnx_options)
            self._encoder = rt.InferenceSession(
                model_files["encoder"],
                **TensorRtOptions.add_profile(onnx_options, self._encoder_shapes),
            )
            self._xattn_kv = rt.InferenceSession(model_files["xattn_kv"], **onnx_options)
            self._decoder_step = rt.InferenceSession(model_files["decoder_step"], **onnx_options)
            # Inherited `_decoding` (deliberately not overridden) reads
            # `self._decoder.get_inputs()` only to shape the initial empty
            # `decoder_mems`; `decoder_step.onnx` exposes the same `decoder_mems`
            # input contract, so aliasing satisfies that lookup without loading
            # the fused decoder.
            self._decoder = self._decoder_step
            self._kv_cache = threading.local()

            # Verbatim from `NemoConformerAED.__init__` (onnx_asr/models/nemo.py).
            self._tokens = {token: token_id for token_id, token in self._vocab.items()}
            self._eos_token_id = self._tokens["<|endoftext|>"]
            # The supported-language set, derived from the loaded vocab rather
            # than hardcoded: whichever `<|xx|>` tokens this checkpoint ships
            # are exactly the languages its decode prefix can request.
            self.language_token_ids = {
                match.group(1): token_id
                for token_id, token in self._vocab.items()
                if (match := _LANGUAGE_TOKEN_RE.fullmatch(token))
            }
            self._transcribe_input = np.array(
                [
                    [
                        self._tokens[" "],
                        self._tokens["<|startofcontext|>"],
                        self._tokens["<|startoftranscript|>"],
                        self._tokens["<|emo:undefined|>"],
                        self._tokens["<|en|>"],
                        self._tokens["<|en|>"],
                        self._tokens["<|pnc|>"],
                        self._tokens["<|noitn|>"],
                        self._tokens["<|notimestamp|>"],
                        self._tokens["<|nodiarize|>"],
                    ]
                ],
                dtype=np.int64,
            )

        # `_get_model_files` is NOT overridden here: `NemoConformerAED` already
        # provides a concrete implementation (satisfying `BaseAsr`'s ABC contract),
        # and this class is never constructed via `onnx_asr.load_model`/`Manager`
        # (which is the only caller of `_get_model_files`) -- the builder function
        # in `onnx_canary_split.py` constructs `model_files` directly against this backend's own
        # artifact directory contract instead.

        def _decode(
            self,
            input_ids: np.ndarray,
            encoder_embeddings: np.ndarray,
            encoder_mask: np.ndarray,
            decoder_mems: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            # The K/V cache lives in a threading.local: each transcribe_pcm call
            # runs its whole decode loop on one asyncio.to_thread worker, and
            # each _decoding call starts with empty decoder_mems so step 0
            # recomputes -- overlapping windows on this shared instance never
            # see each other's tensors.
            if decoder_mems.shape[2] == 0:
                kv_outputs = self._xattn_kv.run(
                    _KV_TENSORS, {"encoder_embeddings": encoder_embeddings}
                )
                self._kv_cache.tensors = {
                    name: np.asarray(arr) for name, arr in zip(_KV_TENSORS, kv_outputs, strict=True)
                }
            kv_cache = getattr(self._kv_cache, "tensors", None)
            assert kv_cache is not None  # noqa: S101 -- set above on step 0 of this thread; every later step reads it

            outputs = self._decoder_step.run(
                ["logits", "decoder_hidden_states"],
                {
                    "input_ids": input_ids if decoder_mems.shape[2] == 0 else input_ids[:, -1:],
                    "encoder_mask": encoder_mask,
                    "decoder_mems": decoder_mems,
                    **kv_cache,
                },
            )
            return np.asarray(outputs[0]), np.asarray(outputs[1])

    return _NemoConformerAEDSplitDecode
