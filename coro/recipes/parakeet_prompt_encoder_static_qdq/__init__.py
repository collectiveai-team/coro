r"""Static QDQ INT8 quantization of the Parakeet-Prompt encoder ONLY.

See ``README.md`` in this package for the full write-up (the
`encoded_lengths` corruption bug this recipe's parameters avoid
re-introducing, and why `decoder_joint` stays fp32). This docstring stays
short on purpose -- it is what ``--help`` shows.

Requires a calibration corpus -- see ``coro.recipes.calibration_corpus``
(``--calibration-manifest`` defaults to that recipe's output location).

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_encoder_static_qdq \\
        --source /path/to/encoder-encoder.onnx
    uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_encoder_static_qdq \\
        --source /path/to/encoder-encoder.onnx \\
        --calibration-manifest recipe-artifacts/calibration_corpus/calibration_manifest.json \\
        --out-dir recipe-artifacts/parakeet_prompt
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import override

import numpy as np
import numpy.typing as npt
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from coro.cli_support import invoke, new_app, out_console
from coro.core.logger import get_logger
from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

log = get_logger(__name__)

# The current-best selector `onnx-parakeet-prompt`'s `quantization` expects
# (`encoder-encoder.<this>.onnx`).
ACCEPTED_SELECTOR = "static_qdq_v3"


def _load_calibration_wavs(manifest_path: Path) -> list[str]:
    """Load calibration clip paths from coro.recipes.calibration_corpus's manifest."""
    if not manifest_path.is_file():
        msg = (
            f"No calibration manifest at {manifest_path} -- run "
            "`python -m coro.recipes.calibration_corpus` first."
        )
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_text())
    return [m["path"] for m in manifest]


type _EncoderFeed = dict[str, npt.NDArray[np.generic]]
"""One calibration input: the encoder's ``audio_signal`` and ``length`` tensors."""


class ParakeetEncoderCalibrationReader(CalibrationDataReader):
    """Feeds real mel-spectrogram features through the encoder's ONNX input contract.

    Input contract is ``audio_signal``/``length``, for MinMax calibration.
    """

    def __init__(self, wav_paths: list[str]) -> None:
        import soundfile as sf

        from onnx_asr.loader import Manager

        manager = Manager(providers=["CPUExecutionProvider"])
        preprocessor = manager._create_preprocessor("nemo80")

        self._inputs: list[_EncoderFeed] = []
        for path in wav_paths:
            audio, sr = sf.read(path, dtype="float32")
            if sr != 16000:
                msg = f"{path} is {sr} Hz, expected 16000"
                raise ValueError(msg)
            waveforms = np.asarray(audio, dtype=np.float32)[None, :]
            waveforms_len = np.array([waveforms.shape[1]], dtype=np.int64)
            features, features_lens = preprocessor(waveforms, waveforms_len)
            self._inputs.append({"audio_signal": features, "length": features_lens})
        self._iter = iter(self._inputs)

    @override
    def get_next(self) -> _EncoderFeed | None:  # pyrefly: ignore[bad-override]
        return next(self._iter, None)


app = new_app("parakeet_prompt_encoder_static_qdq", help=__doc__.splitlines()[0])


@app.default
def _quantize(
    *,
    source: Path,
    calibration_manifest: Path = (
        RECIPE_ARTIFACTS_DIR / "calibration_corpus" / "calibration_manifest.json"
    ),
    out_dir: Path = RECIPE_ARTIFACTS_DIR / "parakeet_prompt",
) -> None:
    """Quantize the Parakeet prompt encoder with static QDQ INT8.

    Args:
        source: fp32 encoder-encoder.onnx to quantize (onnx_asr's own export, not
            produced by any recipe in this package).
        calibration_manifest: Manifest from coro.recipes.calibration_corpus.
        out_dir: Directory to write encoder-encoder.<selector>.onnx into.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessed = out_dir / "encoder-encoder.preprocessed.onnx"
    static_out = out_dir / f"encoder-encoder.{ACCEPTED_SELECTOR}.onnx"

    if preprocessed.exists():
        log.info("pre-processed encoder already cached, reusing", path=str(preprocessed))
    else:
        log.info("pre-processing (shape inference) encoder")
        t0 = time.time()
        quant_pre_process(
            str(source),
            str(preprocessed),
            # >2GB external-data model: ORT's optimization step is unsupported.
            skip_optimization=True,
            # Symbolic shape inference raises "Incomplete symbolic shape
            # inference" on this graph (a `Less` op from NeMo's cache-handling
            # branches confuses it) -- skip it and rely on plain ONNX shape
            # inference instead, which succeeds.
            skip_symbolic_shape=True,
            save_as_external_data=True,
        )
        log.info("pre-processed", seconds=round(time.time() - t0, 1))

    wavs = _load_calibration_wavs(calibration_manifest)
    log.info("building calibration reader", clips=len(wavs))
    reader = ParakeetEncoderCalibrationReader(wavs)

    log.info("running static QDQ quantization (non-VNNI CPU params)")
    t0 = time.time()
    quantize_static(
        model_input=str(preprocessed),
        model_output=str(static_out),
        calibration_data_reader=reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        reduce_range=True,  # recommended on non-VNNI x64
        per_channel=True,
        # Restrict quantization to the actual compute-heavy ops -- see
        # README.md for the encoded_lengths corruption bug this avoids.
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        use_external_data_format=True,
    )
    log.info("quantized", seconds=round(time.time() - t0, 1))
    out_console.out(f"\noutput: {static_out}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the static QDQ encoder quantization recipe."""
    invoke(app, argv)
