"""NeMo ASR Model Integration (nemo backend).

Evaluation vehicle for language-constrained NeMo transducers — primarily the
Parakeet RNNT 1.1B multilingual *Prompt* checkpoints — which the onnx-asr
backend cannot drive today: it has no registry entry for this checkpoint
class, and its ``language`` argument is silently dropped outside its
Whisper/Canary paths. (Earlier notes here claimed the encoder itself needed
an unsupported ``prompt_indices`` input; verified against a real checkpoint
on 2026-09-01, that is wrong -- the encoder is plain and onnx-asr-compatible.
Prompt conditioning is a small *post*-encoder MLP, ``model.prompt_kernel``,
which is why the production path is closer than previously assessed; see
``.scratch/issue-64-language-constrained-asr/findings.md``.)

This adapter runs the PyTorch checkpoint through NeMo's ``transcribe``, where
language forcing works via the request language resolved against the
checkpoint's ``prompt_dictionary`` and passed as ``target_lang``. Token
timestamps come from NeMo's RNNT timestamp computation, so the project-owned
``TranscriptToken`` contract (per-word start/end, chronological order) holds
-- except on the forced-language path, where a NeMo bug forces a fallback
(see the two caveats below).

Two NeMo-level quirks, discovered against a real ``EncDecHybridRNNTCTCBPEModelWithPrompt``
checkpoint (verified 2026-09-01; see ``.scratch/issue-64-language-constrained-asr/findings.md``),
shape how this adapter must call ``transcribe()``:

1. **``target_lang`` as a bare kwarg is silently ignored.** ``transcribe()``'s
   default dataloader (``use_lhotse=True``) reads the target language from a
   manifest supervision's ``language`` field, not from the transcribe config
   -- a bare-file-path call has no such field, so the request is dropped
   without error and the checkpoint falls back to whatever the (missing)
   default resolves to. ``target_lang`` only reaches the model when passed
   via an explicit ``override_config=<TranscribeConfig subclass>(...,
   use_lhotse=False)`` -- only the non-Lhotse dataloader threads
   ``target_lang`` into the per-utterance one-hot prompt tensor. The
   ``override_config`` dataclass type is discovered via introspection
   (:func:`.resolve_override_config_type`) rather than imported by name, so
   this works for any Prompt-conditioned NeMo ASR model class, not only
   ``EncDecHybridRNNTCTCBPEModelWithPrompt``.
2. **``timestamps=True`` crashes on the forced-language path, and even
   ``timestamps=False`` doesn't yield None/empty.** With ``timestamps=True``,
   ``process_timestamp_outputs`` crashes downstream
   (``if 'word' in timestamp`` raises ``RuntimeError: Tensor.__contains__
   only supports Tensor or scalar``) because ``hyp.timestamp`` is a raw
   per-token frame-index ``Tensor`` instead of the expected structured dict.
   With ``timestamps=False`` requested at *both* the top-level kwarg and
   inside ``override_config`` (both are required -- the decoding-strategy
   reset that would normally clear stale timestamp state is gated on the
   top-level kwarg alone, evaluated before ``override_config`` is even
   consulted), the crash is avoided but ``hyp.timestamp`` still comes back
   as that same raw frame-index ``Tensor``, verified directly against the
   checkpoint -- not None, not seconds, not the structured dict.
   :func:`.frame_timestamps_to_seconds` converts these frame indices to
   real per-word timing (``frame_idx * window_stride * subsampling_factor``,
   the same formula NeMo's own ``timestamp_utils.process_timestamp`` uses),
   using conversion factors resolved once at build time by
   :func:`.resolve_frame_conversion_factors` -- verified end-to-end against
   the real checkpoint (5.85s clip, 37 emitted tokens, frame indices
   ``[7, 9, ..., 72]`` -> seconds ``[0.56, 0.72, ..., 5.76]``, all bounded by
   the clip's own duration). When those factors cannot be resolved (an
   unusual checkpoint class lacking ``model.cfg.preprocessor.window_stride``
   or ``model.encoder.subsampling_factor``), the frame indices cannot be
   trusted as seconds, so this adapter falls back to
   :func:`.tokens_from_hypothesis`'s evenly-spaced text timing rather than
   guessing. The auto-detection path (no ``language`` requested) is
   unaffected and keeps NeMo's own timestamps -- though it has not been
   exercised against a real checkpoint this session (this Prompt checkpoint
   has no ``"auto"`` dictionary entry and cannot use it at all); NeMo's
   ``process_timestamp_outputs`` normally returns a structured
   ``{"word": ..., "char": ..., "segment": ...}`` dict on that path, a
   different shape from the forced path's flat frame-index sequence --
   :func:`.frame_timestamps_to_seconds` is never invoked there, and
   :func:`.tokens_from_hypothesis` now also discards a dict-shaped
   ``hyp.timestamp`` defensively (see its docstring) rather than crashing
   iterating over dict keys as floats.

Accepted trade-off: PyTorch eager CPU inference of a 1.1B checkpoint is far
slower than ONNX Runtime with int8 quantisation. This backend exists to
measure *quality* (forced ``es``/``es-US``/``es-ES`` vs auto detection); if
the measurements justify adoption, the production path is an ONNX export
(baked-prompt or upstream ``prompt_indices`` support), not this adapter.

Note:
    Prompt checkpoints whose ``prompt_dictionary`` has no ``"auto"`` entry
    cannot transcribe without an explicit language, so Server Warmup (which
    sends none) fails loudly. Run such checkpoints with ``CORO_WARMUP=disabled``
    until a language-aware warmup exists.

License:
    ``parakeet-rnnt-1.1b-multilingual-prompt`` -- the checkpoint this backend
    exists to drive -- is licensed under the **NVIDIA Community Model
    License**, not extracted as a normal Hugging Face download but from
    inside an NVIDIA NIM container. Production use requires an NVIDIA NIM
    runtime under an NVIDIA AI Enterprise subscription; the license is
    revocable, non-sublicensable, and terminates on any dispute with NVIDIA.
    This is incompatible with distributing coro to run without a dependency
    on an external vendor. Per the same policy already applied to
    ``nvidia/diar_sortformer_4spk-v1`` (CONTEXT.md's Diarization Model
    Selection precedent, itself only CC-BY-NC-4.0 -- weaker restrictions than
    this checkpoint's NIM gate): this backend is a **comparative reference
    only**. It must never be `backend_asr`'s default (``onnx-canary-split`` is,
    see ADR 0019), never recommended to operators, and its weights (or any
    derivative ONNX export of them, e.g. ``onnx-parakeet-prompt``'s
    artifacts) must not be redistributed. See
    ``.scratch/issue-64-language-constrained-asr/findings.md`` for the
    licensing finding that led to the shippable alternative
    (``nvidia/canary-1b-v2``, CC-BY-4.0) that ADR 0019 made the default.

"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.nemo_checkpoint import (
    NemoTranscriber,
    build_forced_language_config,
    resolve_frame_conversion_factors,
    resolve_target_language,
)
from coro.backends.asr.nemo_hypothesis import (
    SubwordTokenizer,
    frame_timestamps_to_seconds,
    tokens_from_timed_hypothesis,
)
from coro.core.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nemo.collections.asr.models import ASRModel

    from coro.core.models import TranscriptToken

logger = get_logger(__name__)

_SAMPLE_RATE = 16000
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32


class NemoASRAdapter:
    """ASRAdapter that wraps a NeMo transducer (EncDecRNNTBPEModel family).

    Adapter Concurrency Policy: **serialised**. A PyTorch module is not
    documented thread-safe for concurrent ``transcribe`` calls, and NeMo's
    decoding mutates model-level decoding state. Serialising is expressed as
    a one-permit :class:`AdmissionController` rather than a ``threading.Lock``,
    so waiting happens on the event loop and overload past the queue-depth cap
    is rejected with a retry hint — the same policy and precedent as the
    onnx-genai backend.
    """

    honours_prompt: bool = False
    """A transducer has no text input port, so the carried prompt is inert.

    Language forcing — the reason this backend exists — is separate: it flows
    through the ``language`` argument and the checkpoint's prompt dictionary,
    not through the carried prompt.
    """

    def __init__(
        self,
        model: NemoTranscriber,
        *,
        tokenizer: SubwordTokenizer | None = None,
        prompt_dictionary: dict[str, int] | None = None,
        frame_conversion: tuple[float, int] | None = None,
        admission: AdmissionController | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._prompt_dictionary = prompt_dictionary
        self._frame_conversion = frame_conversion
        self._admission = admission or build_admission_controller(
            max_concurrency=1, max_queue_depth=_DEFAULT_QUEUE_DEPTH, serialized=True
        )

    @property
    def admission(self) -> AdmissionController:
        """Admission controller implementing this adapter's concurrency policy."""
        return self._admission

    @property
    def frame_conversion(self) -> tuple[float, int] | None:
        """The ``(window_stride, subsampling_factor)`` frame conversion.

        Used to recover true per-word timestamps on the forced-language
        path, or None when the checkpoint didn't expose the needed
        cfg/encoder attributes (see :func:`.resolve_frame_conversion_factors`).
        """
        return self._frame_conversion

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes.

        Note:
            ``prompt`` is accepted for protocol compatibility but ignored (no
            text input port on a transducer).

        Raises:
            AsrCapacityError: If the admission queue is full.
            AsrUnsupportedLanguageError: If an explicit language matches no
                prompt-dictionary key, or the checkpoint carries no prompt
                dictionary at all (so it supports no forced language).
            ValueError: If (forced-language calls only) the checkpoint's
                ``transcribe`` exposes no typed ``override_config`` to force
                the language through -- a NeMo/checkpoint mismatch, which is
                the deployment's fault rather than the request's.

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if language and self._prompt_dictionary is None:
            raise AsrUnsupportedLanguageError(language, supported_languages=())
        target_lang = resolve_target_language(language, self._prompt_dictionary or {})

        duration = len(audio) / _SAMPLE_RATE
        async with self._admission.admit():
            return await asyncio.to_thread(self._transcribe, audio, target_lang, duration)

    def _transcribe(
        self, audio: np.ndarray, target_lang: str | None, duration: float
    ) -> list[TranscriptToken]:
        """Run NeMo ``transcribe`` on a temp wav file and convert the hypothesis.

        NeMo's ``transcribe`` takes file paths, so the window PCM is written to
        a temporary wav (16 kHz, s16le) and removed afterwards; the write is
        negligible next to transducer inference.
        """
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, _SAMPLE_RATE, subtype="PCM_16")
            hypotheses = self._run_transcribe(path, target_lang)
        finally:
            Path(path).unlink(missing_ok=True)

        if not hypotheses:
            return []
        hyp = hypotheses[0]
        raw_timestamp = getattr(hyp, "timestamp", None)
        if target_lang is not None:
            # Even with timestamps=False at both levels, hyp.timestamp still
            # comes back as a raw per-token frame-index Tensor rather than
            # None -- verified directly. Convert it to real per-word seconds
            # here (tokens_from_timed_hypothesis is model-agnostic and has no
            # access to window_stride/subsampling_factor). Falls back to None
            # (-> the evenly-spaced text fallback) when this checkpoint's
            # conversion factors couldn't be resolved at build time.
            raw_timestamp = frame_timestamps_to_seconds(raw_timestamp, self._frame_conversion)
        return tokens_from_timed_hypothesis(hyp, raw_timestamp, self._tokenizer, span_end=duration)

    def _run_transcribe(self, path: str, target_lang: str | None) -> Sequence[object] | None:
        """Call NeMo ``transcribe`` on one wav file, forcing ``target_lang`` when given."""
        if target_lang is None:
            return self._model.transcribe(
                [path],
                timestamps=True,
                return_hypotheses=True,
                batch_size=1,
                verbose=False,
            )
        override_config = build_forced_language_config(self._model, target_lang)
        # `timestamps` must ALSO be passed as a top-level kwarg here,
        # duplicating override_config.timestamps: transcribe()'s
        # decoding-strategy reset (compute_timestamps/preserve_alignments) is
        # gated on the *top-level* `timestamps is not None` check, evaluated
        # before override_config is even consulted. Without it, decoding keeps
        # whatever compute_timestamps was last configured to, and
        # hyp.timestamp comes back as a raw multi-element Tensor (verified
        # against the real checkpoint).
        return self._model.transcribe([path], timestamps=False, override_config=override_config)


def build_nemo_asr_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
) -> NemoASRAdapter:
    """Construct and return a NemoASRAdapter.

    Args:
        model_asr: NeMo model name (``nvidia/parakeet-rnnt-1.1b-...``) or path
            to a local ``.nemo`` checkpoint.
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``); ``auto``
            uses CUDA when torch reports it available.
        max_queue_depth: Calls allowed to wait for the single permit before
            rejection. The permit count is fixed at 1 by this backend's
            Adapter Concurrency Policy, so ``asr_max_concurrency`` does not
            apply.

    Returns:
        Initialised adapter ready for use.

    """
    import nemo.collections.asr as nemo_asr
    import torch

    local = Path(model_asr)
    if local.suffix == ".nemo" and local.exists():
        loaded = nemo_asr.models.ASRModel.restore_from(str(local))
    else:
        loaded = nemo_asr.models.ASRModel.from_pretrained(model_asr)

    resolved = device
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = cast("ASRModel", loaded)  # noqa: BTTOOL004 -- NeMo's loaders are unannotated; their return_config branch infers `| str`, never taken here
    model = asr_model.eval().to(resolved)

    model_defaults = model.cfg.get("model_defaults", {}) or {}
    prompt_dictionary = dict(model_defaults.get("prompt_dictionary", {}) or {}) or None
    # model.tokenizer is NeMo's own TokenizerSpec wrapper (SentencePieceTokenizer,
    # AutoTokenizer, ...), which always implements the abstract `ids_to_tokens`
    # method -- unwrapping to `model.tokenizer.tokenizer` (as this line
    # previously did) reaches into an implementation-specific inner object
    # (the raw `sentencepiece.SentencePieceProcessor` on this checkpoint's
    # SentencePieceTokenizer) that has no such method at all, verified
    # directly against the real checkpoint (AttributeError: 'SentencePieceProcessor'
    # object has no attribute 'convert_ids_to_tokens') -- a latent bug that
    # only surfaced once the forced-language path stopped discarding every
    # timestamp outright.
    tokenizer = cast("SubwordTokenizer", model.tokenizer)  # noqa: BTTOOL004 -- nn.Module.__getattr__ types submodules as `Module | Tensor`
    frame_conversion = resolve_frame_conversion_factors(model)
    if frame_conversion is None:
        logger.warning(
            "Could not resolve window_stride/subsampling_factor for '%s'; "
            "forced-language transcriptions will use evenly-spaced text "
            "timing instead of true per-word timestamps.",
            model_asr,
        )

    logger.info(
        "Loaded NeMo ASR model '%s' (device=%s, prompt_dictionary=%s, frame_conversion=%s).",
        model_asr,
        resolved,
        sorted(prompt_dictionary) if prompt_dictionary else None,
        frame_conversion,
    )
    return NemoASRAdapter(
        cast("NemoTranscriber", model),  # noqa: BTTOOL004 -- ASRModel declares no `transcribe`; the loaded subclass does
        tokenizer=tokenizer,
        prompt_dictionary=prompt_dictionary,
        frame_conversion=frame_conversion,
        admission=build_admission_controller(
            max_concurrency=1, max_queue_depth=max_queue_depth, serialized=True
        ),
    )
