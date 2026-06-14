"""ONNX Runtime GenAI Model Integration (onnx-genai backend).

Wraps NVIDIA Nemotron cache-aware *streaming* ASR models (e.g.
``onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4``) that are exported in the
onnxruntime-genai ``nemotron_speech`` format. These models cannot be driven by onnx-asr
(their encoder threads streaming cache state and needs a ``lang_id`` input), so this backend
uses ``onnxruntime_genai`` directly.

Audio is fed as fixed ``chunk_samples`` (560 ms) float32 chunks to a per-request
``StreamingProcessor`` + ``Generator`` session. Tokens emitted after processing chunk *i*
are timestamped at ``i * chunk_seconds`` (560 ms resolution) -- coarse but sufficient for the
pipeline's segment-level speaker assignment. Inline language-tag tokens (e.g. ``<en-US>``)
are stripped. Word reconstruction + end-time synthesis reuse the onnx-asr converter.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from types import SimpleNamespace

import numpy as np

from asr_diar_server.backends.onnx_asr import convert_onnx_asr_result
from asr_diar_server.core.types import TranscriptToken

logger = logging.getLogger(__name__)

# Language code / locale -> lang_id, per the model card's prompt dictionary.
# "auto" lets the model detect the language.
_LANG_TO_ID = {
    "auto": 101,
    "en": 0, "en-US": 0, "en-GB": 1,
    "es-ES": 2, "es": 3, "es-US": 3,
    "zh": 4, "zh-CN": 4,
    "hi": 6, "ar": 7,
    "fr": 8, "fr-FR": 8, "fr-CA": 100,
    "de": 9, "ja": 10, "ru": 11,
    "pt-BR": 12, "pt": 13, "pt-PT": 13,
    "ko": 14, "it": 15, "nl": 16, "pl": 17,
    "tr": 18, "uk": 19,
}
_DEFAULT_LANG_ID = 0

# Strips inline language-tag tokens like "<en>" or "<en-US>" the model emits.
_LANG_TAG_RE = re.compile(r"<[a-z]{2}(?:-[A-Z]{2})?>")


def _lang_id_for(language: str | None) -> int:
    """Map a language/locale code to the model's lang_id (default English)."""
    if not language:
        return _DEFAULT_LANG_ID
    return _LANG_TO_ID.get(language, _LANG_TO_ID.get(language.split("-")[0], _DEFAULT_LANG_ID))


class OnnxGenaiASRAdapter:
    """ASRAdapter that wraps an onnxruntime-genai nemotron_speech streaming model."""

    def __init__(self, model, *, chunk_samples: int, sample_rate: int) -> None:
        self._model = model
        self._chunk_samples = chunk_samples
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_samples / sample_rate
        self._lock = threading.Lock()

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[TranscriptToken]:
        """Transcribe raw PCM s16le 16 kHz mono bytes via the streaming GenAI pipeline.

        Note:
            ``prompt`` is ignored (no equivalent in the GenAI streaming API). Token
            timestamps have 560 ms resolution (one streaming chunk).

        """
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        lang_id = _lang_id_for(language)

        def _recognize() -> list[TranscriptToken]:
            with self._lock:
                return self._stream(audio, lang_id)

        return await asyncio.to_thread(_recognize)

    def _stream(self, audio: np.ndarray, lang_id: int) -> list[TranscriptToken]:
        """Drive one streaming session over ``audio`` and return reconstructed tokens."""
        import onnxruntime_genai as og

        processor = og.StreamingProcessor(self._model)
        processor.set_option("use_vad", "false")
        tokenizer = og.Tokenizer(self._model)
        tstream = tokenizer.create_stream()
        generator = og.Generator(self._model, og.GeneratorParams(self._model))
        generator.set_runtime_option("lang_id", str(lang_id))

        pieces: list[str] = []
        times: list[float] = []

        def _drain(chunk_time: float) -> None:
            while not generator.is_done():
                generator.generate_next_token()
                toks = generator.get_next_tokens()
                if len(toks) > 0:
                    text = tstream.decode(toks[0])
                    text = _LANG_TAG_RE.sub("", text)
                    if text:
                        pieces.append(text)
                        times.append(chunk_time)

        step = self._chunk_samples
        for i in range(0, len(audio), step):
            chunk = audio[i : i + step].astype(np.float32)
            inputs = processor.process(chunk)
            if inputs is not None:
                generator.set_inputs(inputs)
                _drain(i / self._sample_rate)
        inputs = processor.flush()
        if inputs is not None:
            generator.set_inputs(inputs)
            _drain(len(audio) / self._sample_rate)

        # Reuse the onnx-asr converter for word grouping + end-time synthesis.
        result = SimpleNamespace(tokens=pieces, timestamps=times, logprobs=None)
        return convert_onnx_asr_result(result)


def _apply_device(config, device: str) -> None:
    """Select the execution provider on a GenAI config for the given device.

    onnxruntime-genai ships as separate CPU and CUDA builds; CPU is the implicit
    default when no provider is appended (appending ``"cpu"`` is rejected). ``"auto"``
    follows whatever the model's ``genai_config.json`` declares.
    """
    if device == "cuda":
        config.clear_providers()
        config.append_provider("cuda")
    elif device == "cpu":
        config.clear_providers()


def build_onnx_genai_adapter(
    model_asr: str,
    *,
    device: str = "auto",
    quantization: str | None = None,
) -> OnnxGenaiASRAdapter:
    """Construct and return an OnnxGenaiASRAdapter.

    Args:
        model_asr: HF repo id or local path of an onnxruntime-genai nemotron_speech model.
        device: Device selector (``"auto"``, ``"cuda"``, ``"cpu"``); selects the EP.
        quantization: Unused (the GenAI export already encodes its precision); accepted for
            interface symmetry with the other ASR backends.

    Returns:
        Initialised adapter ready for use.

    """
    import json
    from pathlib import Path

    import onnxruntime_genai as og

    model_dir = Path(model_asr)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(model_asr))
    model_path = str(model_dir)

    with (model_dir / "genai_config.json").open() as fh:
        cfg = json.load(fh)["model"]
    chunk_samples = int(cfg["chunk_samples"])
    sample_rate = int(cfg["sample_rate"])

    logger.info(
        "Loading onnx-genai model '%s' (chunk_samples=%d, device=%s).",
        model_asr,
        chunk_samples,
        device,
    )
    config = og.Config(model_path)
    _apply_device(config, device)
    model = og.Model(config)
    logger.info("onnx-genai model loaded.")
    return OnnxGenaiASRAdapter(model, chunk_samples=chunk_samples, sample_rate=sample_rate)
