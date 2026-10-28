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

from collections.abc import Sequence
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from coro.cli_support import invoke, new_app, out_console
from coro.core.logger import get_logger
from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

log = get_logger(__name__)

# The accepted selector `onnx-canary-split`'s `decoder_quantization` expects
# (`decoder_step.<this>.onnx`) -- see README.md for why QUInt8, not QInt8,
# despite QInt8 looking better on a short-clip microbenchmark.
ACCEPTED_SELECTOR = "dynamic_v1_quint8"


app = new_app("canary_decoder_dynamic_quantization", help=__doc__.splitlines()[0])


@app.default
def _quantize(
    *,
    source: Path = RECIPE_ARTIFACTS_DIR / "canary_split" / "decoder_step.onnx",
    out_dir: Path = RECIPE_ARTIFACTS_DIR / "canary_decoder_int8",
) -> None:
    """Write the dynamically quantized decoder_step variants.

    Args:
        source: fp32 decoder_step.onnx to quantize (coro.recipes.canary_split_decoder's output).
        out_dir: Directory to write the quantized decoder_step.<selector>.onnx variants into.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for weight_type, selector in [
        (QuantType.QUInt8, ACCEPTED_SELECTOR),
        (QuantType.QInt8, "dynamic_v1_qint8"),
    ]:
        out = out_dir / f"decoder_step.{selector}.onnx"
        log.info("quantizing", output=str(out), weight_type=str(weight_type))
        quantize_dynamic(
            model_input=str(source),
            model_output=str(out),
            weight_type=weight_type,
            op_types_to_quantize=["MatMul"],
        )
        log.info("quantized", output=str(out), megabytes=round(out.stat().st_size / 1e6, 1))

    out_console.out(
        f"\nonnx-canary-split's accepted decoder_quantization selector is "
        f"{ACCEPTED_SELECTOR!r} -> decoder_step.{ACCEPTED_SELECTOR}.onnx"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the decoder dynamic quantization recipe."""
    invoke(app, argv)
