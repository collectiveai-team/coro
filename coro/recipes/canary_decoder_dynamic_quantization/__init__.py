r"""Dynamic INT8 quantization of Canary's decoder_step.onnx.

See ``README.md`` in this package for the full write-up (why dynamic
quantization and not static QDQ, and the measured quality/speed numbers).
This docstring stays short on purpose -- it is what ``--help`` shows.

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.canary_decoder_dynamic_quantization
    uv run --extra recipes --extra cpu -m coro.recipes.canary_decoder_dynamic_quantization \\
        --source recipe-artifacts/canary_split/decoder_step.onnx \\
        --out-dir recipe-artifacts/canary_decoder_int8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

# The accepted selector `onnx-canary-split`'s `decoder_quantization` expects
# (`decoder_step.<this>.onnx`) -- see README.md for why QUInt8, not QInt8,
# despite QInt8 looking better on a short-clip microbenchmark.
ACCEPTED_SELECTOR = "dynamic_v1_quint8"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=RECIPE_ARTIFACTS_DIR / "canary_split" / "decoder_step.onnx",
        help="fp32 decoder_step.onnx to quantize (coro.recipes.canary_split_decoder's output).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RECIPE_ARTIFACTS_DIR / "canary_decoder_int8",
        help="Directory to write the quantized decoder_step.<selector>.onnx variants into.",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for weight_type, selector in [
        (QuantType.QUInt8, ACCEPTED_SELECTOR),
        (QuantType.QInt8, "dynamic_v1_qint8"),
    ]:
        out = args.out_dir / f"decoder_step.{selector}.onnx"
        print(f"quantizing -> {out} (weight_type={weight_type})")
        quantize_dynamic(
            model_input=str(args.source),
            model_output=str(out),
            weight_type=weight_type,
            op_types_to_quantize=["MatMul"],
        )
        print(f"  done: {out.stat().st_size / 1e6:.1f} MB")

    print(
        f"\nonnx-canary-split's accepted decoder_quantization selector is "
        f"{ACCEPTED_SELECTOR!r} -> decoder_step.{ACCEPTED_SELECTOR}.onnx"
    )
