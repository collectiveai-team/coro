"""ONNX Canary Split-Decode ASR Model Integration (onnx-canary-split backend).

Wraps a graph-surgery variant of `istupakov/canary-1b-v2-onnx`'s decoder that
hoists cross-attention key/value computation out of the per-token decode loop.
See `.scratch/canary-decode-loop-rtf/PRD.md` (issue #64) and
`.scratch/issue-64-language-constrained-asr/findings.md` for the full
investigation this backend implements.

`decoder-model.onnx` has no cross-attention K/V cache: 16 nodes (8 decoder
layers x {key, value} cross-attention projections) consume the encoder's
output directly and are recomputed unchanged on every decode step, even
though their result never changes within a window. `coro/recipes/canary_split_decoder/`
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

Artifact directory contract (``model_asr`` is a directory or a Hugging Face
repo id -- never a single file), following the same convention
``onnx-parakeet-prompt`` established for the directory case and
``onnx-genai`` for the local-path-else-``snapshot_download`` resolution:

- ``encoder-model.onnx`` (+ ``.onnx.data`` external-data sidecar, when
  present): fp32 encoder, unmodified from `istupakov/canary-1b-v2-onnx`.
  ``encoder-model.<quantization>.onnx`` is loaded instead when ``quantization``
  is given. The accepted INT8 selector is ``static_qdq_v4_pct_excl``, produced
  by ``coro/recipes/canary_encoder_static_qdq/`` (see that package's
  README.md): full 48-window mTEDx validation put it at norm cpWER 0.0508 vs
  the fp32 encoder's 0.0513 in the same run, at +4.0% RTFx. Ticket 04's first
  attempt at this graph (``static_qdq_v3``, MinMax activation calibration,
  every Conv/MatMul/Gemm quantized) was **rejected** at +34.5% relative cpWER,
  and percentile calibration *alone* still failed at +19.9% -- what closes the
  gap is excluding the 32 nodes an ``onnxruntime.quantization.qdq_loss_debug``
  sensitivity pass measured as worst (overwhelmingly the last eight conformer
  layers' convolution module). Do not "simplify" that exclusion list away.
- ``xattn_kv.onnx``: the cross-attention K/V graph from
  ``coro/recipes/canary_split_decoder/`` (see that package's README.md).
  Never quantized -- it runs once per window, not once per decode step, so
  it is not a meaningful cost target.
- ``decoder_step.onnx``: the per-token decode graph from the same split.
  ``decoder_step.<decoder_quantization>.onnx`` is loaded instead when
  ``decoder_quantization`` is given. Ticket 04's *static-QDQ* INT8 attempt on
  this graph was rejected (real word-level WER damage, e.g. "fell-Americans"
  for "fellow Americans" -- see
  ``.journals/2026-09-04/2026-09-04_canary-decode-loop-rtf_decoder-quant-static-qdq-rejected-plus-research/``).
  Dynamic quantization (``onnxruntime.quantization.quantize_dynamic``,
  ``QuantType.QUInt8``, MatMul-only, no calibration data) is the technique
  that actually works here -- full 48-window mTEDx validation: norm cpWER
  0.0529 vs the fp32 decoder's 0.0513 (+3.1% relative, an order of magnitude
  smaller than ticket 04's rejected encoder-INT8 cost of +34.5%), RTFx 2.33
  vs the fp32 split-decode baseline's 1.62 (+43.8%). See
  ``.journals/2026-09-04/2026-09-04_canary-decode-loop-rtf_decoder-dynamic-quantization-accepted/``
  for the full experiment (including why static QDQ specifically fails on an
  autoregressive decoder, and why a QInt8 variant that looked *better* on a
  short-clip microbenchmark actually came in slower than fp32 at full scale
  -- do not trust short-clip numbers alone for this graph). The artifact this
  selector expects, ``decoder_step.dynamic_v1_quint8.onnx``, is produced by
  ``coro/recipes/canary_decoder_dynamic_quantization/`` (see that package's
  README.md).
- ``vocab.txt``: onnx_asr's own ``<token> <id>`` format.
- ``config.json``: optional; ``max_sequence_length`` etc, same as plain
  ``onnx-asr``'s Canary config.

Hugging Face resolution (when ``model_asr`` is not an existing local
directory): the repo id is resolved with ``huggingface_hub.snapshot_download``
using ``allow_patterns`` derived from the selected quantizations, so only the
needed files are pulled -- the accepted default INT8/INT8 selection
(``static_qdq_v4_pct_excl`` + ``dynamic_v1_quint8``) downloads exactly
``encoder-model.static_qdq_v4_pct_excl.onnx(+.data)``,
``decoder_step.dynamic_v1_quint8.onnx``, ``xattn_kv.onnx``, ``vocab.txt`` and
``config.json`` (~1.29 GB), never the 609 MB fp32 ``decoder_step.onnx`` and
never an encoder that was not selected. The default repo is
`collectiveai/canary-1b-v2-onnx-split-int8`, which holds every artifact above
**except the fp32 encoder**.

Two-repo fp32-encoder rule (this backend only): the fp32 encoder selector --
``quantization=None``, or the explicit ``"fp32"`` sentinel -- is not in the
default repo, so it resolves ``encoder-model.onnx`` +
``encoder-model.onnx.data`` from a second ``snapshot_download`` of
`istupakov/canary-1b-v2-onnx` (the unmodified upstream export the local
artifact directories already symlink to). The two repos land in different
cache snapshots, so the builder assembles ``model_files`` from both resolved
paths rather than assuming one directory. Note that ``None`` therefore *means
fp32* here, unlike backends whose unquantized graph lives in the same repo;
``decoder_quantization=None`` still means the fp32 ``decoder_step.onnx``,
which *does* live in the default repo. A Hugging Face token from
``ServerSettings.hf_token`` (``CORO_HF_TOKEN``/``HF_TOKEN``) is forwarded to
every download. A file still missing after its download raises the same
``FileNotFoundError`` shape as the local case, with the repo id and filename
in the message.

Forced-language hardening: every decode runs in an explicitly resolved
language. ``resolve_language`` reduces a request language (``es-US``,
``es_US``, `` ES ``) to a base subtag and checks it against the ``<|xx|>``
tokens actually present in the loaded vocab (``language_token_ids``, derived
at load -- never a hardcoded list); an unsupported language raises
:class:`AsrUnsupportedLanguageError` at the adapter call boundary, which every
request surface maps the way ``AsrCapacityError`` is mapped. A request
carrying no language decodes with ``asr_fallback_language``
(``CORO_ASR_FALLBACK_LANGUAGE``, default ``en``) -- the same resolution
Server Warmup passes explicitly -- *unless* auto-LID (below) resolves one
first; the pipeline layer decides which applies, never this adapter.

Auto-LID (ticket 05): this checkpoint *can* predict its source language --
see ``.scratch/canary-default/findings-lid-probe.md`` (ticket 01's probe:
100% accuracy across 64 speech windows). :meth:`OnnxCanarySplitASRAdapter.detect_language`
exposes it as a standalone call (module-level ``_partial_prompt_lid``): two
greedy decoder steps on NeMo's Canary2 ``user_partial`` prefix
(``<|startofcontext|><|startoftranscript|>``), returning the second step's
``<|xx|>`` token when this checkpoint's vocab carries one, else ``None``. It
runs its own encoder pass rather than reusing a concurrent transcription
call's -- a deliberate simplicity-over-throughput trade documented on that
method. Not part of the :class:`~coro.core.protocols.ASRAdapter` protocol;
every pipeline duck-types it (``getattr(asr, "detect_language", None)``), so
every other backend is unaffected, and both stack-wrapping layers this
backend can sit under (:class:`~coro.cache.adapter.CachingASRAdapter`,
:class:`~coro.backends.asr.factory.LazyASRAdapter`) forward it. Request-scoped
sticky state (the first window whose detection succeeds fixes the language
for the rest of the request) lives in ``coro/pipelines/windowing.py``'s
``LanguageState``, not here -- this adapter is a shared, concurrent instance
that must not carry per-request state. All three pipelines (Full-Memory,
Streaming, Live) drive it, each with its own per-request/per-connection
``LanguageState``.

Adapter Concurrency Policy: **concurrent**. ``_decode``'s split-graph override
caches the 16 cross-attention K/V tensors (computed on the first step of each
window, read on every subsequent step of the *same* ``_decoding`` call) in a
``threading.local()`` holder rather than plain instance state: every
``transcribe_pcm`` call runs its whole ``recognize_batch`` on one worker thread
via ``asyncio.to_thread``, so per-thread storage makes overlapping windows on
one shared instance race-free without any locking. Each ``_decoding`` call
starts with empty ``decoder_mems``, so the first step per thread recomputes the
K/V tensors -- a later window on the same thread never reuses an earlier
window's cache. ONNX Runtime ``InferenceSession.run`` is thread-safe, so the
three sessions themselves stay shared. Load is bounded by an
:class:`AdmissionController` sized the same way ``onnx-asr`` sizes its own:
permit count from ``asr_max_concurrency`` (0 auto-sizes from the core count).

License:
    `nvidia/canary-1b-v2` is **CC-BY-4.0** -- unlike `nemo`/`onnx-parakeet-prompt`
    (both driving `parakeet-rnnt-1.1b-multilingual-prompt`, NVIDIA Community
    Model License, NIM-gated), it carries no redistribution or NIM/AI-Enterprise
    production-use restriction. Do not copy those backends' license-comment
    pattern onto this one -- it does not apply.

    This backend (Canary-1b-v2, INT8 encoder + INT8 decoder) is the **default**
    ASR Backend Provider (`--model-asr canary-1b-v2`, the default slug) -- see
    ADR 0019 for why: the previous default's implicit, uncontrollable per-frame
    language identification is disqualifying for single-language recordings,
    and this backend's forced-language and sticky-auto-LID design (above) fixes
    that at an accepted RTFx cost. `onnx-asr` (`parakeet-tdt-0.6b-v3`) remains
    available by slug for deployments that value raw throughput over language
    stability -- see docs/benchmark.md.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.nemo import resolve_target_language
from coro.backends.asr.onnx_asr import convert_onnx_asr_result
from coro.backends.asr.onnx_session import build_asr_session_options
from coro.cache.fingerprint import normalise_language
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
_DECODER_STEP_BASENAME = "decoder_step"
_VOCAB_FILENAME = "vocab.txt"
_CONFIG_FILENAME = "config.json"

# Default HF repo holding the whole contract except the fp32 encoder, and the
# upstream export the fp32 encoder (and the split graphs' source) came from.
# See the module docstring's Hugging Face resolution section.
_DEFAULT_SPLIT_REPO = "collectiveai/canary-1b-v2-onnx-split-int8"
_FP32_ENCODER_REPO = "istupakov/canary-1b-v2-onnx"
# Explicit sentinel for "no quantization"; the slug registry (ticket 06)
# normalises it at the settings layer, this builder accepts it directly too.
_FP32_SELECTOR = "fp32"


def _is_fp32(quantization: str | None) -> bool:
    """Whether a quantization selector names the unquantized fp32 graph."""
    return quantization is None or quantization == _FP32_SELECTOR


# Matches exactly the two-letter language switch tokens in Canary's vocab
# (`<|en|>`, `<|es|>`, ...) -- deliberately narrower than any `<|...|>` token,
# so control tokens (`<|pnc|>`, `<|noitn|>`, `<|emo:...|>`) are excluded and
# the supported set is whatever this checkpoint's vocab actually ships.
_LANGUAGE_TOKEN_RE = re.compile(r"<\|([a-z]{2})\|>")


def resolve_canary_language(language: str | None, language_tokens: dict[str, int]) -> str | None:
    """Resolve a request language to a Canary ``<|xx|>`` vocab code.

    Composed from the two normalisers the codebase already has (no third one
    is added): :func:`coro.cache.fingerprint.normalise_language` handles the
    spelling (strip / lowercase / underscore-to-hyphen), then NeMo's
    :func:`~coro.backends.asr.nemo.resolve_target_language` does the
    exact-then-primary-subtag match against the vocab-derived token map --
    so ``es-US``, ``es_US`` and ``" ES "`` all resolve to ``es``.

    Args:
        language: Request language, or None/blank to leave unresolved (the
            caller applies its fallback).
        language_tokens: The ``code -> token id`` map derived from the loaded
            vocab; also the supported set the error names.

    Returns:
        The base-subtag code whose ``<|xx|>`` token keys the decode prefix.

    Raises:
        AsrUnsupportedLanguageError: If the language matches no ``<|xx|>``
            token in the loaded vocab.

    """
    normalised = normalise_language(language)
    try:
        return resolve_target_language(normalised, language_tokens)
    except ValueError as exc:
        raise AsrUnsupportedLanguageError(normalised, supported_languages=language_tokens) from exc


def _partial_prompt_lid(asr: Any, pcm: bytes) -> str | None:
    """Run NeMo's Canary2 partial-prompt LID probe on one PCM window.

    NeMo's ``Canary2PromptFormatter`` documents a ``user_partial`` role
    (``<|startofcontext|><|startoftranscript|>``) used for exactly two
    decoder steps to retrieve the emotion and source-language tokens.
    ``.scratch/canary-default/findings-lid-probe.md`` (ticket 01) measured
    this against the split-decode INT8/INT8 checkpoint: 100% accuracy across
    64 probed speech windows (48 mTEDx es, 4 each en/fr/de/pt), step 1 always
    ``<|emo:undefined|>`` (no emotion capability), step 2 the language.

    Deliberately runs its own encoder pass rather than reusing a concurrent
    transcription call's -- keeping :meth:`OnnxCanarySplitASRAdapter.detect_language`
    a self-contained call (no `transcribe_pcm` return-type/contract change,
    see that method's docstring) costs one extra encoder pass on the *first*
    one or two windows of a request needing detection, never a sustained
    per-window cost once :class:`~coro.pipelines.windowing.LanguageState`
    goes sticky.

    Args:
        asr: The loaded split-decode ``NemoConformerAED`` subclass instance.
        pcm: Raw PCM s16le 16 kHz mono bytes for one window.

    Returns:
        The emitted language code (e.g. ``"es"``) when the second step's
        token matches this checkpoint's ``<|xx|>`` language-token shape, else
        ``None`` -- not observed on real audio in the probe (even silence
        emitted some language token), but handled defensively.

    """
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    waveforms = audio[None, :]
    waveforms_len = np.array([len(audio)], dtype=np.int64)
    features, features_lens = asr._preprocessor(waveforms, waveforms_len)
    encoder_embeddings, encoder_mask = asr._encode(features, features_lens)

    prefix = np.array(
        [
            [
                asr._tokens[" "],
                asr._tokens["<|startofcontext|>"],
                asr._tokens["<|startoftranscript|>"],
            ]
        ],
        dtype=np.int64,
    )
    shapes = {x.name: x.shape for x in asr._decoder.get_inputs()}
    decoder_mems = np.empty(
        (shapes["decoder_mems"][0], 1, 0, shapes["decoder_mems"][3]), dtype=np.float32
    )

    batch_tokens = prefix
    next_token_id = -1
    for _ in range(2):
        logits, decoder_mems = asr._decode(
            batch_tokens, encoder_embeddings, encoder_mask, decoder_mems
        )
        next_token_id = int(np.argmax(logits[:, -1], axis=-1)[0])
        batch_tokens = np.concatenate([batch_tokens, [[next_token_id]]], axis=-1)

    match = _LANGUAGE_TOKEN_RE.fullmatch(asr._vocab[next_token_id])
    return match.group(1) if match else None


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
    if _is_fp32(quantization):
        return f"{_ENCODER_BASENAME}.onnx"
    return f"{_ENCODER_BASENAME}.{quantization}.onnx"


def _decoder_step_filename(decoder_quantization: str | None) -> str:
    """Return the decoder_step ONNX filename for a decoder quantization selector."""
    if _is_fp32(decoder_quantization):
        return f"{_DECODER_STEP_BASENAME}.onnx"
    return f"{_DECODER_STEP_BASENAME}.{decoder_quantization}.onnx"


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
            self._tokens = {token: id for id, token in self._vocab.items()}
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
        # below constructs `model_files` directly against this backend's own
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


class OnnxCanarySplitASRAdapter:
    """ASRAdapter wrapping the split-decode Canary pipeline.

    Adapter Concurrency Policy: **concurrent** -- the split-decode K/V cache is
    thread-local and the ONNX Runtime sessions are shared and thread-safe; see
    the module docstring. Load is bounded by an :class:`AdmissionController`
    sized from ``asr_max_concurrency`` (0 = auto-size from the core count).
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
        fallback_language: str = "en",
    ) -> None:
        self._asr = asr
        # The ASR instance's vocab-derived `<|xx|>` map -- this checkpoint's
        # supported-language set, never a hardcoded list.
        self._language_tokens: dict[str, int] = asr.language_token_ids
        self._fallback_language = fallback_language
        self._admission = admission or build_admission_controller(
            max_concurrency=0, max_queue_depth=_DEFAULT_QUEUE_DEPTH
        )

    @property
    def admission(self) -> AdmissionController:
        """Admission controller implementing this adapter's concurrency policy."""
        return self._admission

    @property
    def supported_languages(self) -> frozenset[str]:
        """Language codes this checkpoint can decode with, derived from its vocab."""
        return frozenset(self._language_tokens)

    @property
    def fallback_language(self) -> str:
        """Language used when a request carries none (``asr_fallback_language``).

        Public so the ASR Window Cache decorator can key an absent request
        language the same way :meth:`resolve_language` decodes it (base
        subtag of this value), without the cache layer knowing anything else
        about this backend -- see ``coro/cache/adapter.py``.
        """
        return self._fallback_language

    def resolve_language(self, language: str | None) -> str:
        """Resolve a request language to a vocab-backed Canary code.

        A blank or absent request language resolves to the adapter's
        ``fallback_language`` (``asr_fallback_language`` via the factory), so
        the decode prefix is never left to onnx_asr's hardcoded ``<|en|>``.
        Detection (slice 05) will take precedence by passing a detected
        language explicitly -- explicit beats the fallback here.

        Raises:
            AsrUnsupportedLanguageError: If the requested language (or the
                fallback, when the request carries none) matches no ``<|xx|>``
                token in the loaded vocab.

        """
        requested = language if (language or "").strip() else self._fallback_language
        resolved = resolve_canary_language(requested, self._language_tokens)
        if resolved is None:  # blank request and blank fallback
            raise AsrUnsupportedLanguageError("", supported_languages=self._language_tokens)
        return resolved

    async def detect_language(self, pcm: bytes) -> str | None:
        """Detect one window's source language via the Canary2 partial-prompt LID probe.

        Not part of the :class:`~coro.core.protocols.ASRAdapter` protocol --
        the auto-LID sticky pipeline layer (``coro/pipelines/windowing.py``)
        duck-types this (``getattr(asr, "detect_language", None)``), so every
        other backend is unaffected.

        Returns:
            The base-subtag language code (e.g. ``"es"``) when the probe's
            second decoder step emitted one of this checkpoint's ``<|xx|>``
            language tokens, else ``None``. Cross-checked against
            ``self._language_tokens`` (the same vocab-derived set
            :meth:`resolve_language` uses) rather than trusting
            :func:`_partial_prompt_lid`'s own regex match in isolation.

        """

        def _detect() -> str | None:
            code = _partial_prompt_lid(self._asr, pcm)
            return code if code in self._language_tokens else None

        async with self._admission.admit():
            return await asyncio.to_thread(_detect)

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
            AsrUnsupportedLanguageError: If the language (or the fallback,
                when the request carries none) matches no ``<|xx|>`` vocab
                token. Raised at resolution time, before admission, so every
                request surface gets it for free.

        """
        resolved = self.resolve_language(language)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / _SAMPLE_RATE

        def _recognize():
            waveforms = audio[None, :]
            waveforms_len = np.array([len(audio)], dtype=np.int64)
            return next(
                iter(self._asr.recognize_batch(waveforms, waveforms_len, language=resolved))
            )

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


def _require_file(directory: Path, filename: str, *, repo_id: str | None = None) -> Path:
    """Return the artifact path, raising this backend's FileNotFoundError shape.

    ``repo_id`` is included in the message when resolution came from the hub,
    so a missing post-download file names the repo and the filename.
    """
    path = directory / filename
    if not path.is_file():
        suffix = f" (repo {repo_id})" if repo_id else ""
        msg = f"Missing required onnx-canary-split artifact: {path}{suffix}"
        raise FileNotFoundError(msg)
    return path


@dataclass(frozen=True)
class _SplitArtifacts:
    """Resolved artifact paths for one build (local directory or HF snapshots).

    The encoder and the split-decode graphs may live in different HF cache
    snapshots (fp32-encoder rule -- see the module docstring), which is why
    resolution yields named paths rather than one directory.
    """

    encoder: Path
    xattn_kv: Path
    decoder_step: Path
    vocab: Path
    config: Path | None = None


def _optional_config(directory: Path) -> Path | None:
    """Return ``config.json`` when present (optional in both local and hub contracts)."""
    config_path = directory / _CONFIG_FILENAME
    return config_path if config_path.is_file() else None


def _artifacts_from_directory(
    directory: Path, *, quantization: str | None, decoder_quantization: str | None
) -> _SplitArtifacts:
    """Resolve the artifact contract against a local artifact directory.

    Raises:
        FileNotFoundError: If a required artifact is missing (path in the message).

    """
    return _SplitArtifacts(
        encoder=_require_file(directory, _encoder_filename(quantization)),
        xattn_kv=_require_file(directory, _XATTN_KV_FILENAME),
        decoder_step=_require_file(directory, _decoder_step_filename(decoder_quantization)),
        vocab=_require_file(directory, _VOCAB_FILENAME),
        config=_optional_config(directory),
    )


def _artifacts_from_hub(
    repo_id: str,
    *,
    quantization: str | None,
    decoder_quantization: str | None,
    hf_token: str | None,
) -> _SplitArtifacts:
    """Resolve the artifact contract from HF snapshots, downloading only what is needed.

    ``allow_patterns`` are derived from the selected quantizations so the
    default INT8/INT8 selection pulls ~1.29 GB, never the 609 MB fp32
    ``decoder_step.onnx``. The fp32 encoder is absent from the split repos, so
    selecting it means a second ``snapshot_download`` from
    ``_FP32_ENCODER_REPO`` and resolved paths spanning two snapshots -- see
    the module docstring's Hugging Face resolution section.

    Raises:
        FileNotFoundError: If a required artifact is still missing after the
            download, with the repo id and filename in the message.

    """
    from huggingface_hub import snapshot_download

    encoder_name = _encoder_filename(quantization)
    shared_patterns = [
        _decoder_step_filename(decoder_quantization),
        _XATTN_KV_FILENAME,
        _VOCAB_FILENAME,
        _CONFIG_FILENAME,
    ]
    if _is_fp32(quantization):
        encoder_repo_id = _FP32_ENCODER_REPO
        encoder_snapshot = Path(
            snapshot_download(
                encoder_repo_id,
                allow_patterns=[encoder_name, f"{encoder_name}.data"],
                token=hf_token,
            )
        )
        split_snapshot = Path(
            snapshot_download(repo_id, allow_patterns=shared_patterns, token=hf_token)
        )
    else:
        encoder_repo_id = repo_id
        encoder_snapshot = split_snapshot = Path(
            snapshot_download(
                repo_id,
                allow_patterns=[encoder_name, f"{encoder_name}.data", *shared_patterns],
                token=hf_token,
            )
        )

    return _SplitArtifacts(
        encoder=_require_file(encoder_snapshot, encoder_name, repo_id=encoder_repo_id),
        xattn_kv=_require_file(split_snapshot, _XATTN_KV_FILENAME, repo_id=repo_id),
        decoder_step=_require_file(
            split_snapshot, _decoder_step_filename(decoder_quantization), repo_id=repo_id
        ),
        vocab=_require_file(split_snapshot, _VOCAB_FILENAME, repo_id=repo_id),
        config=_optional_config(split_snapshot),
    )


def _resolve_artifacts(
    model_asr: str,
    *,
    quantization: str | None,
    decoder_quantization: str | None,
    hf_token: str | None,
) -> _SplitArtifacts:
    """Resolve the artifact contract to concrete paths: local directory else HF hub.

    Follows ``onnx-genai``'s resolution pattern: an existing local directory
    is used as-is (no hub call at all); anything else is a Hugging Face repo
    id. The default repo is ``_DEFAULT_SPLIT_REPO``, but it must be named
    explicitly in ``model_asr`` -- flipping the default is ticket 06.
    """
    directory = Path(model_asr)
    if directory.is_dir():
        return _artifacts_from_directory(
            directory, quantization=quantization, decoder_quantization=decoder_quantization
        )
    return _artifacts_from_hub(
        repo_id=model_asr,
        quantization=quantization,
        decoder_quantization=decoder_quantization,
        hf_token=hf_token,
    )


def build_onnx_canary_split_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    quantization: str | None = None,
    decoder_quantization: str | None = None,
    providers: Sequence[str] | None = None,
    max_concurrency: int = 0,
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
    hf_token: str | None = None,
    fallback_language: str = "en",
) -> OnnxCanarySplitASRAdapter:
    """Construct and return an OnnxCanarySplitASRAdapter.

    Args:
        model_asr: Local directory holding the artifact contract described in
            this module's docstring, or a Hugging Face repo id (default repo:
            ``collectiveai/canary-1b-v2-onnx-split-int8``) resolved via
            ``snapshot_download`` with only the selected quantizations'
            files -- see the module docstring's Hugging Face resolution
            section for the two-repo fp32-encoder rule.
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``) used to
            derive providers when ``providers`` is not given explicitly.
        quantization: Encoder quantization selector (e.g.
            ``"static_qdq_v4_pct_excl"``, the accepted static-QDQ INT8 variant
            -- see the module docstring for the two earlier variants of it that
            were rejected); ``None`` (or the ``"fp32"`` sentinel) loads the
            fp32 encoder from ``istupakov/canary-1b-v2-onnx`` when resolving
            from the hub -- this backend's documented exception to
            ``None``-means-in-the-same-repo.
        decoder_quantization: ``decoder_step.onnx`` quantization selector
            (e.g. ``"dynamic_v1_quint8"``, the accepted dynamic-INT8 variant
            -- see the module docstring for why static QDQ was rejected for
            this graph); ``None`` (or ``"fp32"``) loads the fp32 decoder step.
        providers: Explicit onnxruntime providers; overrides ``device`` when supplied.
        max_concurrency: Adapter Concurrency Policy permit count; 0 auto-sizes
            from the host core count.
        max_queue_depth: Calls allowed to wait for a permit before rejection.
        hf_token: Hugging Face token (``ServerSettings.hf_token``, read from
            ``CORO_HF_TOKEN``/``HF_TOKEN``); forwarded to ``snapshot_download``
            and ignored for local directories.
        fallback_language: Language used when a request carries none
            (``ServerSettings.asr_fallback_language``). This checkpoint has no
            auto-detection, so the fallback -- not a hidden ``en`` inside
            onnx_asr's prefix -- is what a no-language request decodes with.

    Returns:
        Initialised adapter ready for use.

    Raises:
        FileNotFoundError: If a required artifact is missing from ``model_asr``
            (local) or still missing after the download (hub; message carries
            the repo id and filename).

    """
    from onnx_asr.loader import Manager

    artifacts = _resolve_artifacts(
        model_asr,
        quantization=quantization,
        decoder_quantization=decoder_quantization,
        hf_token=hf_token,
    )
    # ``model_files`` dict keys are onnx_asr's own constructor contract.
    model_files: dict[str, Path] = {
        "encoder": artifacts.encoder,
        "xattn_kv": artifacts.xattn_kv,
        "decoder_step": artifacts.decoder_step,
        "vocab": artifacts.vocab,
    }
    if artifacts.config is not None:
        model_files["config"] = artifacts.config

    resolved_providers = providers if providers is not None else _providers_for_device(device)
    session_options = build_asr_session_options()
    logger.info(
        "Loading onnx-canary-split model from '%s' (quantization=%s, "
        "decoder_quantization=%s, providers=%s).",
        model_asr,
        quantization,
        decoder_quantization,
        resolved_providers,
    )
    manager = Manager(sess_options=session_options, providers=resolved_providers)
    asr_cls = _split_canary_asr_class()
    asr = asr_cls(model_files, manager._create_preprocessor, manager.default_onnx_config)
    logger.info("onnx-canary-split model loaded.")
    return OnnxCanarySplitASRAdapter(
        asr,
        admission=build_admission_controller(
            max_concurrency=max_concurrency, max_queue_depth=max_queue_depth
        ),
        fallback_language=fallback_language,
    )
