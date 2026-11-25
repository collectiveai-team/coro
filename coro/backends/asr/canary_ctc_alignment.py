"""Word alignment with the CTC model bundled in NVIDIA's Canary checkpoint."""

from __future__ import annotations

import tarfile
from math import isfinite
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Protocol

from coro.core.logger import get_logger
from coro.core.models import TranscriptToken

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

CANARY_CTC_REVISION = "d455706339a6b32e1aa40f82c713a482a0c938e2"
_ARCHIVE_NAME = "canary-1b-v2.nemo"
_CONFIG = "timestamps_asr_model_config.yaml"
_WEIGHTS = "timestamps_asr_model_weights.ckpt"
# The auxiliary config refers to these archive entries by their full names.
_ARTIFACTS = {
    _CONFIG,
    _WEIGHTS,
    "241aa2c5e2bf4e448725c04a1ee5a00b_tokenizer.model",
    "5640ba46de3f4c699d991b1fd9bde0d4_vocab.txt",
    "2a7ae133a5024390b7554d587d0cf637_tokenizer.vocab",
}
_SAMPLE_RATE = 16000
logger = get_logger(__name__)


class CanaryWordAligner(Protocol):
    """Align Canary's words to the original audio window."""

    def align(self, audio: npt.NDArray[np.float32], text: str) -> list[TranscriptToken]:
        """Return the original words with start and end times in seconds."""
        ...


class NemoCanaryWordAligner:
    """Use NeMo's forced alignment without loading NeMo's Canary decoder."""

    def __init__(self, model: object) -> None:
        self._model = model
        self._lock = Lock()

    def align(self, audio: npt.NDArray[np.float32], text: str) -> list[TranscriptToken]:
        """Align one window; NeMo's transcribe flow mutates shared model state."""
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        from nemo.collections.asr.parts.utils.timestamp_utils import (
            get_forced_aligned_timestamps_with_external_model,
        )

        hypothesis = Hypothesis(score=0.0, y_sequence=[], text=text)
        with self._lock:
            get_forced_aligned_timestamps_with_external_model(
                # NeMo accepts waveform lists but annotates only path lists.
                audio=[audio],  # pyrefly: ignore[bad-argument-type]
                external_ctc_model=self._model,
                main_model_predictions=[hypothesis],
                batch_size=1,
                timestamp_type="word",
                verbose=False,
            )
        timestamps = hypothesis.timestamp
        if not isinstance(timestamps, dict) or "word" not in timestamps:
            raise ValueError("Canary CTC alignment did not return word times")
        return _aligned_words(text, timestamps["word"], len(audio) / _SAMPLE_RATE)


def _aligned_words(
    text: str, stamps: list[dict[str, str | float]], duration: float
) -> list[TranscriptToken]:
    """Reject partial or invalid matches instead of silently changing the transcript."""
    words = text.split()
    if words != [stamp["word"] for stamp in stamps]:
        raise ValueError("Canary CTC alignment did not match all transcript words")
    tokens: list[TranscriptToken] = []
    previous_end = 0.0
    for word, stamp in zip(words, stamps, strict=True):
        start, end = float(stamp["start"]), float(stamp["end"])
        if not (isfinite(end) and previous_end <= start < end and start < duration):
            raise ValueError("Canary CTC alignment returned invalid word times")
        # The last encoder frame may extend past the end of the audio.
        end = min(end, duration)
        tokens.append(TranscriptToken(start=start, end=end, text=" " + word))
        previous_end = end
    return tokens


def _extract_ctc(archive: str, destination: str) -> None:
    """Extract the bundled aligner config, weights and tokenizer artifacts."""
    with tarfile.open(archive, mode="r:*") as bundle:
        members = [
            member
            for member in bundle.getmembers()
            if member.isfile() and member.name.removeprefix("./") in _ARTIFACTS
        ]
        names = {member.name.removeprefix("./") for member in members}
        if missing := _ARTIFACTS - names:
            raise ValueError(f"Canary checkpoint is missing CTC artifacts: {sorted(missing)}")
        bundle.extractall(destination, members=members, filter="data")


def _restore_ctc(directory: str, device: str) -> object:
    """Restore only the auxiliary CTC model from the extracted directory."""
    import torch
    from nemo.collections.asr.models import ASRModel
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    connector = SaveRestoreConnector()
    connector.model_extracted_dir = directory
    connector.model_config_yaml = _CONFIG
    connector.model_weights_ckpt = _WEIGHTS
    model = ASRModel.restore_from(
        restore_path=directory,
        map_location=torch.device("cpu"),
        save_restore_connector=connector,
    )
    return model.eval().to(device)


def _alignment_device(providers: Sequence[str] | None) -> str:
    """Match the CPU or CUDA execution providers enabled for Canary."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    enabled = set(providers or available).intersection(available)
    return "cuda" if enabled & {"CUDAExecutionProvider", "TensorrtExecutionProvider"} else "cpu"


def build_canary_word_aligner(
    *, providers: Sequence[str] | None, hf_token: str | None
) -> CanaryWordAligner:
    """Load the pinned NVIDIA CTC aligner once at server startup."""
    from huggingface_hub import hf_hub_download

    device = _alignment_device(providers)
    logger.info("Loading Canary CTC aligner from NVIDIA's checkpoint (device=%s).", device)
    archive = hf_hub_download(
        repo_id="nvidia/canary-1b-v2",
        filename=_ARCHIVE_NAME,
        revision=CANARY_CTC_REVISION,
        token=hf_token,
    )
    with TemporaryDirectory(prefix="coro-canary-ctc-") as directory:
        _extract_ctc(archive, directory)
        model = _restore_ctc(directory, device)
    return NemoCanaryWordAligner(model)
