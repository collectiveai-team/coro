"""ONNX Parakeet-Prompt ASR Model Integration (onnx-parakeet-prompt backend).

Wraps the quantized, standalone ONNX export of the NeMo Parakeet RNNT 1.1B
Multilingual *Prompt* checkpoint (``EncDecHybridRNNTCTCBPEModelWithPrompt``) --
the same checkpoint the ``nemo`` backend drives via PyTorch, but through
``onnxruntime`` directly so a static-QDQ INT8 encoder can be used on CPU. See
``.scratch/issue-64-language-constrained-asr/findings.md`` for the
quantization-strategy research (static QDQ, encoder-only,
``op_types_to_quantize=["Conv", "MatMul", "Gemm"]``, per-channel + reduce_range)
this adapter's artifacts are expected to have been produced by.

``onnx_asr`` (the pip package) has no registry entry for this checkpoint class,
and its transducer decode loop has no text input port for prompt conditioning,
so this module subclasses its ``onnx_asr.models.nemo.NemoConformerRnnt``
directly, reusing its RNNT greedy-decode loop, preprocessing and vocabulary
handling unmodified, and overriding only ``_encode`` to run the checkpoint's
``model.prompt_kernel`` MLP (``Linear(hidden+num_prompts, 2*hidden) -> ReLU ->
Linear(2*hidden, hidden)``) on the encoder's output before decoding -- forcing
the requested language via a per-timestep one-hot vector concatenated onto the
encoder's hidden state. That MLP is small enough that findings.md's research
scripts (and this adapter) replicate it in NumPy from the checkpoint's
``state_dict()`` rather than exporting a second ONNX graph.

Artifact directory contract (``model_asr`` is a directory, not a single file).
Filenames match ``coro/recipes/parakeet_prompt_encoder_static_qdq/``'s and
``coro/recipes/parakeet_prompt_kernel_cache/``'s output exactly (see each
package's README.md, ``coro/recipes/__init__.py``, and
``.scratch/issue-64-language-constrained-asr/``) rather than a new
convention invented here, so the ~1.1 GB of already-exported artifacts
never need regenerating or renaming to be usable:

- ``encoder-encoder.onnx``: fp32 encoder. ``encoder-encoder.<quantization>.onnx``
  (+ a same-named ``.onnx.data`` external-data sidecar, when present) is loaded
  instead when ``quantization`` is given -- e.g. ``"static_qdq_v3"`` for the
  current-best static-QDQ INT8 encoder.
- ``decoder_joint-encoder.onnx``: fp32 decoder_joint. Never quantized regardless
  of ``quantization`` -- findings.md's static-QDQ research restricts quantization
  to the encoder only; decoder_joint is small and called once per emitted token,
  so quantizing it risks decode-loop degradation for negligible size savings (a
  quantized ``decoder_joint-encoder.int8.onnx`` may exist alongside it from that
  research; this adapter never loads it).
- ``vocab.txt``: one ``<token> <id>`` pair per line (onnx_asr's own format),
  terminated by a ``<blk> <id>`` line naming the blank token id.
- ``prompt_kernel_cache.npz``: the checkpoint's ``model.prompt_kernel.state_dict()``
  arrays (``0.weight``, ``0.bias``, ``2.weight``, ``2.bias``).
- ``prompt_kernel_cache.json``: ``{"prompt_dictionary": {<language>: <prompt_id>},
  ...}`` (extra keys such as ``vocab_size``/``blank_id`` are ignored -- ``vocab.txt``
  is this adapter's single source of truth for both).

Concurrency: see :class:`OnnxParakeetPromptASRAdapter`'s docstring.

License:
    This backend's artifacts are a derivative ONNX export of
    ``parakeet-rnnt-1.1b-multilingual-prompt``, licensed under the **NVIDIA
    Community Model License** and gated behind an NVIDIA NIM runtime / AI
    Enterprise subscription for production use -- see ``coro/backends/asr/nemo.py``'s
    module docstring for the full license summary and CONTEXT.md precedent
    this policy follows. **Comparative reference only**: never the default
    (`onnx-asr` is, and stays), never recommended, and these exported weights
    must not be uploaded/redistributed -- doing so would defeat the NIM/AI-
    Enterprise production gate the source license exists to enforce. The
    graph-surgery/quantization *code* that produced them has no such
    restriction and may be published freely; only the resulting weights are
    constrained.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from coro.backends.asr.concurrency import AdmissionController, build_admission_controller
from coro.backends.asr.errors import AsrUnsupportedLanguageError
from coro.backends.asr.nemo_checkpoint import resolve_target_language
from coro.backends.asr.onnx_asr import convert_onnx_asr_result
from coro.backends.asr.onnx_parakeet_prompt_kernel import (
    load_prompt_kernel,
    prompt_conditioned_asr_class,
)
from coro.backends.asr.onnx_session import build_asr_session_options
from coro.core.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from contextlib import AbstractContextManager

    import numpy.typing as npt

    from coro.core.models import TranscriptToken

logger = get_logger(__name__)

_SAMPLE_RATE = 16000
# Admission queue depth used when an adapter is built without explicit settings
# (direct construction in tests and tooling); the factory always passes one.
_DEFAULT_QUEUE_DEPTH = 32

# Artifact directory contract -- see module docstring. Names match the
# existing export tooling's output exactly (not a convention invented here).
_ENCODER_BASENAME = "encoder-encoder"
_DECODER_JOINT_FILENAME = "decoder_joint-encoder.onnx"
_VOCAB_FILENAME = "vocab.txt"


class PromptConditionedAsr(Protocol):
    """The prompt-conditioned ``NemoConformerRnnt`` surface the adapter calls."""

    def forced_prompt(self, prompt_id: int) -> AbstractContextManager[None]:
        """Force ``prompt_id`` for one ``recognize_batch`` call."""
        ...

    def recognize_batch(
        self,
        waveforms: npt.NDArray[np.float32],
        waveforms_len: npt.NDArray[np.int64],
        /,
        **kwargs: object,
    ) -> Iterator[object]:
        """Recognize a waveform batch, one result per waveform."""
        ...


def _encoder_filename(quantization: str | None) -> str:
    """Return the encoder ONNX filename for a quantization selector."""
    if quantization:
        return f"{_ENCODER_BASENAME}.{quantization}.onnx"
    return f"{_ENCODER_BASENAME}.onnx"


class OnnxParakeetPromptASRAdapter:
    """ASRAdapter wrapping the quantized ONNX Parakeet-Prompt pipeline.

    Adapter Concurrency Policy: **serialised**. Forcing a language means
    setting the underlying ASR instance's ``_prompt_id`` for the duration of
    one ``recognize_batch`` call (see :class:`.ForcedPrompt`) -- mutable
    instance state that two overlapping calls would race on, even though the
    underlying ONNX Runtime sessions are themselves documented thread-safe.
    Serialising is expressed as a one-permit :class:`AdmissionController`, the
    same policy and precedent as the ``nemo``/``onnx-genai`` backends.
    """

    honours_prompt: bool = False
    """A transducer has no text input port, so the carried prompt is inert.

    Language forcing flows through the ``language`` argument and the
    checkpoint's prompt dictionary, not through the carried prompt -- the same
    convention as the ``nemo`` backend, which forces this same checkpoint via
    PyTorch instead of ONNX Runtime.
    """

    def __init__(
        self,
        asr: PromptConditionedAsr,
        *,
        prompt_dictionary: dict[str, int],
        admission: AdmissionController | None = None,
    ) -> None:
        self._asr = asr
        self._prompt_dictionary = prompt_dictionary
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
        """Transcribe raw PCM s16le 16 kHz mono bytes with a forced language.

        Note:
            ``prompt`` is accepted for protocol compatibility but ignored (no
            text input port on a transducer).

        Raises:
            AsrCapacityError: If the admission queue is full.
            AsrUnsupportedLanguageError: If no language is given -- this
                checkpoint's prompt dictionary has no "auto" entry (see
                findings.md), so a forced language is always required -- or the
                given language matches no prompt-dictionary key.

        """
        if not language:
            raise AsrUnsupportedLanguageError("", supported_languages=self._prompt_dictionary)
        target_lang = resolve_target_language(language, self._prompt_dictionary)
        assert target_lang is not None  # noqa: S101 -- `language` is truthy, so resolve_target_language cannot return None
        prompt_id = self._prompt_dictionary[target_lang]

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / _SAMPLE_RATE

        def _recognize() -> object:
            waveforms = audio[None, :]
            waveforms_len = np.array([len(audio)], dtype=np.int64)
            with self._asr.forced_prompt(prompt_id):
                return next(
                    iter(self._asr.recognize_batch(waveforms, waveforms_len, need_logprobs=True))
                )

        async with self._admission.admit():
            result = await asyncio.to_thread(_recognize)
        return convert_onnx_asr_result(result, span_end=duration)


def _providers_for_device(device: str) -> Sequence[str] | None:
    """Map an ASR device selector to onnxruntime execution providers.

    Duplicated from ``onnx_asr.py`` rather than imported: adapter modules do
    not import each other's private helpers (this project already extracted
    ``subword_tokens.py`` for the one piece of logic that needed sharing --
    see ``nemo.py``'s module docstring).
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


def build_onnx_parakeet_prompt_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    quantization: str | None = None,
    providers: Sequence[str] | None = None,
    max_queue_depth: int = _DEFAULT_QUEUE_DEPTH,
) -> OnnxParakeetPromptASRAdapter:
    """Construct and return an OnnxParakeetPromptASRAdapter.

    Args:
        model_asr: Path to a directory holding the artifact contract described
            in this module's docstring (encoder/decoder_joint ONNX files,
            vocab.txt, prompt_kernel.npz/.json).
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``) used to
            derive providers when ``providers`` is not given explicitly.
        quantization: Encoder quantization selector (e.g. ``"static_qdq_v3"``);
            ``None`` loads the fp32 ``encoder.onnx``. decoder_joint is always
            fp32 -- see the module docstring.
        providers: Explicit onnxruntime providers; overrides ``device`` when supplied.
        max_queue_depth: Calls allowed to wait for the single permit before
            rejection. The permit count is fixed at 1 by this backend's
            Adapter Concurrency Policy, so ``asr_max_concurrency`` does not apply.

    Returns:
        Initialised adapter ready for use.

    Raises:
        FileNotFoundError: If a required artifact is missing from ``model_asr``.

    """
    from onnx_asr.loader import Manager

    directory = Path(model_asr)
    encoder_path = directory / _encoder_filename(quantization)
    decoder_joint_path = directory / _DECODER_JOINT_FILENAME
    vocab_path = directory / _VOCAB_FILENAME
    for path in (encoder_path, decoder_joint_path, vocab_path):
        if not path.is_file():
            msg = f"Missing required onnx-parakeet-prompt artifact: {path}"
            raise FileNotFoundError(msg)

    prompt_weights, prompt_dictionary = load_prompt_kernel(directory)

    resolved_providers = providers if providers is not None else _providers_for_device(device)
    session_options = build_asr_session_options()
    logger.info(
        "Loading onnx-parakeet-prompt model from '%s' (quantization=%s, providers=%s).",
        model_asr,
        quantization,
        resolved_providers,
    )
    manager = Manager(sess_options=session_options, providers=resolved_providers)
    asr_cls = prompt_conditioned_asr_class()
    asr = asr_cls(
        {"encoder": encoder_path, "decoder_joint": decoder_joint_path, "vocab": vocab_path},
        manager._create_preprocessor,
        manager.default_onnx_config,
        prompt_kernel_weights=prompt_weights,
    )
    logger.info(
        "onnx-parakeet-prompt model loaded (prompt_dictionary=%s).", sorted(prompt_dictionary)
    )
    return OnnxParakeetPromptASRAdapter(
        asr,
        prompt_dictionary=prompt_dictionary,
        admission=build_admission_controller(
            max_concurrency=1, max_queue_depth=max_queue_depth, serialized=True
        ),
    )
