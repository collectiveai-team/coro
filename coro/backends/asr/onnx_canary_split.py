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

import logging
from typing import TYPE_CHECKING

from coro.backends.asr.concurrency import build_admission_controller
from coro.backends.asr.onnx_canary_split_adapter import (
    _DEFAULT_QUEUE_DEPTH,
    OnnxCanarySplitASRAdapter,
)
from coro.backends.asr.onnx_canary_split_artifacts import _resolve_artifacts
from coro.backends.asr.onnx_canary_split_decoder import _split_canary_asr_class
from coro.backends.asr.onnx_canary_split_language import resolve_canary_language
from coro.backends.asr.onnx_session import build_asr_session_options

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "OnnxCanarySplitASRAdapter",
    "build_onnx_canary_split_adapter",
    "resolve_canary_language",
]

logger = logging.getLogger(__name__)


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
