"""ONNX Canary Split-Decode ASR Model Integration (onnx-canary-split backend).

Wraps a graph-surgery variant of `istupakov/canary-1b-v2-onnx`'s decoder that
hoists cross-attention key/value computation out of the per-token decode loop.
See `.scratch/canary-decode-loop-rtf/PRD.md` (issue #64) and
`.scratch/issue-64-language-constrained-asr/findings.md` for the full
investigation this backend implements.

`decoder-model.onnx` has no cross-attention K/V cache: 16 nodes (8 decoder
layers x {key, value} cross-attention projections) consume the encoder's
output directly and are recomputed unchanged on every decode step, even
though their result never changes within a window. `.tmp/split_canary_decoder.py`
cuts the fused graph in two via `onnx.utils.extract_model`
(`xattn_kv.onnx` computes the 16 K/V tensors once per window; `decoder_step.onnx`
takes them as extra inputs alongside `input_ids`/`encoder_mask`/`decoder_mems`)
and verifies the split composes back to the fused graph at
`max|Δ| = 0.000e+00`. This module drives those two split graphs instead of the
fused decoder.

``onnx_asr`` (the pip package) has no split-decoder-graph concept, so this
module subclasses its ``onnx_asr.models.nemo.NemoConformerAED`` directly,
following the same integration pattern ``onnx-parakeet-prompt`` already uses
(subclass + override the decode path only) -- see
``coro/backends/asr/onnx_parakeet_prompt.py``'s module docstring for that
precedent. Unlike that backend, this subclass overrides ``__init__`` fully
rather than extending it: ``NemoConformerAED.__init__`` unconditionally loads
the ~645 MB fused ``decoder-model.onnx``, which this backend never calls, so
loading it would be pure waste. ``_encode``, preprocessing, tokenization, and
``_decoding``'s greedy-decode loop (prefix construction, EOS handling,
``max_sequence_length``) are all inherited unmodified -- only ``_decode`` is
overridden, per the PRD's integration-seam note.

Artifact directory contract (``model_asr`` is a directory, not a single file),
following the same convention ``onnx-parakeet-prompt`` already established:

- ``encoder-model.onnx`` (+ ``.onnx.data`` external-data sidecar, when
  present): fp32 encoder, unmodified from `istupakov/canary-1b-v2-onnx`.
  ``encoder-model.<quantization>.onnx`` is loaded instead when ``quantization``
  is given (e.g. the encoder INT8 static-QDQ variant from
  ``.tmp/quantize_canary_encoder.py`` -- see the PRD's ticket 04; the selector
  is plumbed through here regardless of whether that artifact exists yet, per
  the PRD's append-only shared-surface convention).
- ``xattn_kv.onnx`` / ``decoder_step.onnx``: the two split-decoder graphs from
  ``.tmp/split_canary_decoder.py``. Never quantized -- ticket 04 quantizes
  only the encoder.
- ``vocab.txt``: onnx_asr's own ``<token> <id>`` format.
- ``config.json``: optional; ``max_sequence_length`` etc, same as plain
  ``onnx-asr``'s Canary config.

Adapter Concurrency Policy: **serialised**. ``_decode``'s split-graph override
caches the 16 cross-attention K/V tensors as mutable instance state
(``self._kv_cache``, computed on the first step of each window and read on
every subsequent step of the *same* ``_decoding`` call) -- two overlapping
``recognize_batch`` calls on the same instance would race on it, the same
category of problem ``onnx-parakeet-prompt``'s ``_prompt_id`` has (see that
module's docstring). Serialising via a one-permit :class:`AdmissionController`
is the same precedent this project already uses for that reason. A
thread-local cache would let this be concurrent instead (each
``asyncio.to_thread`` call runs entirely on one thread), but this backend is
comparative-reference status only, so the extra complexity is not justified
without a concrete throughput need.

License:
    `nvidia/canary-1b-v2` is **CC-BY-4.0** -- unlike `nemo`/`onnx-parakeet-prompt`
    (both driving `parakeet-rnnt-1.1b-multilingual-prompt`, NVIDIA Community
    Model License, NIM-gated), it carries no redistribution or NIM/AI-Enterprise
    production-use restriction. Do not copy those backends' license-comment
    pattern onto this one -- it does not apply. Still **comparative reference
    only**: never the default (`onnx-asr` is, and stays), never recommended --
    see the PRD's non-goals (this program exists to fix Canary's RTF, not to
    replace `onnx-asr` as the default).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.onnx_asr import convert_onnx_asr_result
from coro.backends.asr.onnx_session import build_asr_session_options
from coro.core.models import TranscriptToken

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32

# Artifact directory contract -- see module docstring. Names match
# `.tmp/split_canary_decoder.py`'s and the upstream `canary-1b-v2-onnx`
# repo's own filenames exactly (not a convention invented here).
_ENCODER_BASENAME = "encoder-model"
_XATTN_KV_FILENAME = "xattn_kv.onnx"
_DECODER_STEP_FILENAME = "decoder_step.onnx"
_VOCAB_FILENAME = "vocab.txt"
_CONFIG_FILENAME = "config.json"

# The 16 cross-attention K/V frontier tensors -- see `.tmp/split_canary_decoder.py`'s
# module docstring for why these tensors (not the shallower MatMul outputs) are
# the maximal hoistable cut. Duplicated here (rather than imported) because that
# script lives in `.tmp/`, which this package must not depend on at import time.
_KV_TENSORS: list[str] = []
for _layer in range(8):
    _KV_TENSORS.append(f"/_decoder/layers.{_layer}/second_sub_layer/Transpose_3_output_0")
    _KV_TENSORS.append(f"/_decoder/layers.{_layer}/second_sub_layer/Transpose_2_output_0")


def _encoder_filename(quantization: str | None) -> str:
    """Return the encoder ONNX filename for a quantization selector."""
    if quantization:
        return f"{_ENCODER_BASENAME}.{quantization}.onnx"
    return f"{_ENCODER_BASENAME}.onnx"


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
            self._kv_cache: dict[str, np.ndarray] | None = None

            # Verbatim from `NemoConformerAED.__init__` (onnx_asr/models/nemo.py).
            self._tokens = {token: id for id, token in self._vocab.items()}
            self._eos_token_id = self._tokens["<|endoftext|>"]
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
        # below constructs `model_files` directly against this backend's own
        # artifact directory contract instead.

        def _decode(
            self,
            input_ids: np.ndarray,
            encoder_embeddings: np.ndarray,
            encoder_mask: np.ndarray,
            decoder_mems: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            if decoder_mems.shape[2] == 0:
                kv_outputs = self._xattn_kv.run(
                    _KV_TENSORS, {"encoder_embeddings": encoder_embeddings}
                )
                self._kv_cache = {
                    name: np.asarray(arr) for name, arr in zip(_KV_TENSORS, kv_outputs, strict=True)
                }
            assert self._kv_cache is not None  # noqa: S101 -- set above on step 0; every later step reads it

            outputs = self._decoder_step.run(
                ["logits", "decoder_hidden_states"],
                {
                    "input_ids": input_ids if decoder_mems.shape[2] == 0 else input_ids[:, -1:],
                    "encoder_mask": encoder_mask,
                    "decoder_mems": decoder_mems,
                    **self._kv_cache,
                },
            )
            return np.asarray(outputs[0]), np.asarray(outputs[1])

    return _NemoConformerAEDSplitDecode


class OnnxCanarySplitASRAdapter:
    """ASRAdapter wrapping the split-decode Canary pipeline.

    Adapter Concurrency Policy: **serialised** -- see the module docstring.
    """

    honours_prompt: bool = False
    """Canary's decoder has no carried-prompt input port (only `language`/`pnc`
    prefix tokens); this matches `onnx-asr`'s own Canary integration.
    """

    def __init__(
        self,
        asr: Any,
        *,
        admission: AdmissionController | None = None,
    ) -> None:
        self._asr = asr
        self._admission = admission or build_admission_controller(
            max_concurrency=1, max_queue_depth=_DEFAULT_QUEUE_DEPTH, serialized=True
        )

    @property
    def admission(self) -> AdmissionController:
        """Admission controller implementing this adapter's concurrency policy."""
        return self._admission

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes with an optional forced language.

        Note:
            ``prompt`` is accepted for protocol compatibility but ignored --
            same as plain ``onnx-asr``'s Canary integration: no text input
            port for a carried prompt, only `language`/`target_language`/`pnc`
            prefix tokens.

        Raises:
            AsrCapacityError: If the admission queue is full.

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / _SAMPLE_RATE

        def _recognize():
            waveforms = audio[None, :]
            waveforms_len = np.array([len(audio)], dtype=np.int64)
            kwargs: dict = {}
            if language:
                kwargs["language"] = language
            return next(iter(self._asr.recognize_batch(waveforms, waveforms_len, **kwargs)))

        async with self._admission.admit():
            result = await asyncio.to_thread(_recognize)
        # No per-token timestamps for this AED model (same as `onnx-asr`'s Canary
        # and Whisper integrations) -- `convert_onnx_asr_result` already falls
        # back to `words_from_text` when `result.timestamps` is None, which
        # `_decoding`'s `None` second yield element guarantees here too. Do not
        # regress or paper over that known limitation (see findings.md).
        return convert_onnx_asr_result(result, span_end=duration)


def _providers_for_device(device: str) -> Sequence[str] | None:
    """Map an ASR device selector to onnxruntime execution providers.

    Duplicated from ``onnx_asr.py`` rather than imported: adapter modules do
    not import each other's private helpers (see ``onnx_parakeet_prompt.py``'s
    module docstring for the same convention).
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


def build_onnx_canary_split_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    quantization: str | None = None,
    providers: Sequence[str] | None = None,
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
) -> OnnxCanarySplitASRAdapter:
    """Construct and return an OnnxCanarySplitASRAdapter.

    Args:
        model_asr: Path to a directory holding the artifact contract described
            in this module's docstring (encoder + split-decoder-graph ONNX
            files, vocab.txt, optional config.json).
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``) used to
            derive providers when ``providers`` is not given explicitly.
        quantization: Encoder quantization selector (e.g. an encoder INT8
            static-QDQ variant); ``None`` loads the fp32 encoder. The split
            decoder graphs are never quantized -- see the module docstring.
        providers: Explicit onnxruntime providers; overrides ``device`` when supplied.
        max_queue_depth: Calls allowed to wait for the single permit before
            rejection. The permit count is fixed at 1 by this backend's
            Adapter Concurrency Policy.

    Returns:
        Initialised adapter ready for use.

    Raises:
        FileNotFoundError: If a required artifact is missing from ``model_asr``.

    """
    from onnx_asr.loader import Manager

    directory = Path(model_asr)
    encoder_path = directory / _encoder_filename(quantization)
    xattn_kv_path = directory / _XATTN_KV_FILENAME
    decoder_step_path = directory / _DECODER_STEP_FILENAME
    vocab_path = directory / _VOCAB_FILENAME
    for path in (encoder_path, xattn_kv_path, decoder_step_path, vocab_path):
        if not path.is_file():
            msg = f"Missing required onnx-canary-split artifact: {path}"
            raise FileNotFoundError(msg)

    model_files: dict[str, Path] = {
        "encoder": encoder_path,
        "xattn_kv": xattn_kv_path,
        "decoder_step": decoder_step_path,
        "vocab": vocab_path,
    }
    config_path = directory / _CONFIG_FILENAME
    if config_path.is_file():
        model_files["config"] = config_path

    resolved_providers = providers if providers is not None else _providers_for_device(device)
    session_options = build_asr_session_options()
    logger.info(
        "Loading onnx-canary-split model from '%s' (quantization=%s, providers=%s).",
        model_asr,
        quantization,
        resolved_providers,
    )
    manager = Manager(sess_options=session_options, providers=resolved_providers)
    asr_cls = _split_canary_asr_class()
    asr = asr_cls(model_files, manager._create_preprocessor, manager.default_onnx_config)
    logger.info("onnx-canary-split model loaded.")
    return OnnxCanarySplitASRAdapter(
        asr,
        admission=build_admission_controller(
            max_concurrency=1, max_queue_depth=max_queue_depth, serialized=True
        ),
    )
