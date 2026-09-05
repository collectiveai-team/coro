r"""Static QDQ INT8 quantization of the Canary encoder ONLY.

See ``README.md`` in this package for the full write-up: why the first
attempt (MinMax calibration, every Conv/MatMul/Gemm quantized) was rejected,
what `onnxruntime.quantization.qdq_loss_debug` measured, and how the two
changes here -- percentile activation calibration and a measured
`nodes_to_exclude` list -- recover the loss. This docstring stays short on
purpose: it is what ``--help`` shows.

Requires a calibration corpus -- see ``coro.recipes.calibration_corpus``
(``--calibration-manifest`` defaults to that recipe's output location).

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.canary_encoder_static_qdq \\
        --source /path/to/encoder-model.onnx
    uv run --extra recipes --extra cpu -m coro.recipes.canary_encoder_static_qdq \\
        --source /path/to/encoder-model.onnx \\
        --calibration-manifest recipe-artifacts/calibration_corpus/calibration_manifest.json \\
        --out-dir recipe-artifacts/canary_split
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

# The selector `onnx-canary-split`'s `quantization` argument expects
# (`encoder-model.<this>.onnx`), and `Settings.asr_quantization` selects.
ACCEPTED_SELECTOR = "static_qdq_v4_pct_excl"

# Activation calibration percentile. The rejected v3 attempt used MinMax,
# which stretches every scale to the single most extreme activation seen
# across the corpus; clipping at 99.999% instead lifted the median per-tensor
# quantization SNR from 32.7 dB to 34.0 dB and the accumulated cross-model SNR
# from 10.6 dB to 13.7 dB. See README.md.
CALIBRATION_PERCENTILE = 99.999

# Histogram-based calibration (Percentile/Entropy) must see exactly one clip
# per collection chunk: ORT's HistogramCollector does `np.asarray()` over a
# whole chunk, which raises on an inhomogeneous shape because calibration
# clips have different durations and therefore different frame counts.
CALIBRATION_STRIDE = 1

# The 32 nodes whose quantization noise `qdq_loss_debug` measured as worst
# (2-22 dB SNR, against a ~34 dB median) in the percentile-calibrated graph,
# left in fp32. Overwhelmingly the last conformer layers' convolution module
# (`conv1d_74`..`conv1d_95` are the layer-24..31 pointwise projections, the
# `node_Conv_39xx`/`node_Conv_40xx` block their depthwise partners) plus the
# late-layer attention/feed-forward MatMuls. Derived by
# `.tmp/canary_encoder_qdq_sensitivity.py` and pinned here rather than
# re-measured at build time so this recipe is deterministic -- regenerate the
# list with that script if the upstream fp32 encoder ever changes.
NODES_TO_EXCLUDE: list[str] = [
    "node_conv1d_86", "node_conv1d_83", "node_conv1d_80", "node_conv1d_89",
    "node_conv1d_77", "node_MatMul_2921", "node_MatMul_3019", "node_MatMul_3117",
    "node_conv1d_92", "node_MatMul_3215", "node_MatMul_3308", "node_MatMul_3296",
    "node_MatMul_1451", "node_conv1d_95", "node_conv1d_74", "node_Conv_3732",
    "node_Conv_4007", "node_Conv_4018", "node_Conv_3996", "node_MatMul_3014",
    "node_Conv_3952", "node_Conv_4029", "node_Conv_3963", "node_MatMul_170",
    "node_Conv_3776", "node_MatMul_3210", "node_Conv_4040", "node_Conv_4062",
    "node_conv2d_3", "node_Conv_3754", "node_Conv_3787", "node_MatMul_2916",
]  # fmt: skip


def _load_calibration_wavs(manifest_path: Path) -> list[str]:
    """Load calibration clip paths from coro.recipes.calibration_corpus's manifest.

    Falls back to `<manifest_dir>/<lang>/<basename>` for entries whose recorded
    absolute path no longer resolves: the manifest records where the clips were
    written, which does not survive moving the corpus between machines or
    worktrees, but the corpus layout itself does.
    """
    if not manifest_path.is_file():
        msg = (
            f"No calibration manifest at {manifest_path} -- run "
            "`python -m coro.recipes.calibration_corpus` first."
        )
        raise FileNotFoundError(msg)
    manifest = json.loads(manifest_path.read_text())

    wavs: list[str] = []
    for entry in manifest:
        recorded = Path(entry["path"])
        if recorded.is_file():
            wavs.append(str(recorded))
            continue
        relocated = manifest_path.parent / entry["lang"] / recorded.name
        if not relocated.is_file():
            msg = f"Calibration clip missing at both {recorded} and {relocated}"
            raise FileNotFoundError(msg)
        wavs.append(str(relocated))
    return wavs


class CanaryEncoderCalibrationReader(CalibrationDataReader):
    """Feeds real mel-spectrogram features through the encoder's ONNX input contract.

    Input contract is ``audio_signal``/``length`` (the encoder never sees
    ``target_lang`` -- language only enters at the decoder prefix).

    Also implements the ``__len__``/``set_range`` protocol ORT's chunked
    calibration path (``CalibStridedMinMax``) drives, so histogram-based
    calibration does not have to hold every clip's intermediate tensors at
    once -- which OOMs on a 1 B-parameter encoder.
    """

    def __init__(self, wav_paths: list[str]) -> None:
        import soundfile as sf

        from onnx_asr.loader import Manager

        manager = Manager(providers=["CPUExecutionProvider"])
        preprocessor = manager._create_preprocessor("nemo128")

        self._inputs: list[dict] = []
        for path in wav_paths:
            audio, sr = sf.read(path, dtype="float32")
            if sr != 16000:
                msg = f"{path} is {sr} Hz, expected 16000"
                raise ValueError(msg)
            waveforms = np.asarray(audio, dtype=np.float32)[None, :]
            waveforms_len = np.array([waveforms.shape[1]], dtype=np.int64)
            features, features_lens = preprocessor(waveforms, waveforms_len)
            self._inputs.append({"audio_signal": features, "length": features_lens})
        self._start = 0
        self._end = len(self._inputs)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._inputs)

    def set_range(self, start_index: int, end_index: int) -> None:
        """Restrict `get_next` to `[start_index, end_index)` (ORT chunking hook)."""
        self._start = start_index
        self._end = min(end_index, len(self._inputs))
        self._cursor = start_index

    def get_next(self) -> dict | None:  # pyrefly: ignore[bad-override]
        if self._cursor >= self._end:
            return None
        item = self._inputs[self._cursor]
        self._cursor += 1
        return item

    def rewind(self) -> None:
        self._cursor = self._start


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="fp32 encoder-model.onnx to quantize (istupakov/canary-1b-v2-onnx's "
        "own export, not produced by any recipe in this package).",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=RECIPE_ARTIFACTS_DIR / "calibration_corpus" / "calibration_manifest.json",
        help="Manifest from coro.recipes.calibration_corpus.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RECIPE_ARTIFACTS_DIR / "canary_split",
        help="Directory to write encoder-model.<selector>.onnx into.",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preprocessed = args.out_dir / "encoder-model.preprocessed.onnx"
    static_out = args.out_dir / f"encoder-model.{ACCEPTED_SELECTOR}.onnx"
    # Activation ranges depend on the corpus and calibration method only, never
    # on which nodes are excluded -- caching them makes a rebuild with a
    # different exclusion list minutes rather than tens of minutes.
    calibration_cache = args.out_dir / "encoder-model.calibration_percentile.json"

    if preprocessed.exists():
        print(
            f"pre-processed encoder already cached at {preprocessed}, "
            "reusing (fp32 source unchanged)"
        )
    else:
        print("pre-processing (shape inference) encoder...")
        t0 = time.time()
        quant_pre_process(
            str(args.source),
            str(preprocessed),
            # >2GB external-data model: ORT's optimization step is unsupported.
            skip_optimization=True,
            # Symbolic shape inference chokes on this graph; plain ONNX shape
            # inference succeeds. Same fix the Parakeet encoder recipe needs.
            skip_symbolic_shape=True,
            save_as_external_data=True,
        )
        print(f"  done in {time.time() - t0:.1f}s")

    wavs = _load_calibration_wavs(args.calibration_manifest)
    print(f"building calibration reader ({len(wavs)} clips)...")
    reader = CanaryEncoderCalibrationReader(wavs)

    print(
        f"running static QDQ quantization (percentile={CALIBRATION_PERCENTILE}, "
        f"{len(NODES_TO_EXCLUDE)} nodes kept in fp32, non-VNNI CPU params)..."
    )
    t0 = time.time()
    quantize_static(
        model_input=str(preprocessed),
        model_output=str(static_out),
        calibration_data_reader=reader,
        calibrate_method=CalibrationMethod.Percentile,
        calibration_cache_path=str(calibration_cache),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        reduce_range=True,  # recommended on non-VNNI x64
        per_channel=True,
        # Restrict quantization to the actual compute-heavy ops -- see
        # README.md for the encoder_mask corruption bug this avoids.
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        nodes_to_exclude=list(NODES_TO_EXCLUDE),
        use_external_data_format=True,
        extra_options={
            "CalibStridedMinMax": CALIBRATION_STRIDE,
            "CalibPercentile": CALIBRATION_PERCENTILE,
        },
    )
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"\noutput: {static_out}")
