"""coro.recipes.canary_decoder_dynamic_quantization: selector naming and argparse wiring.

``onnxruntime.quantization.quantize_dynamic`` itself is mocked -- these
tests protect this recipe's own contract (which selectors it produces,
which weight type each maps to, default paths), not onnxruntime's own
quantization behavior (exercised for real when the recipe is actually run;
see the recipe's README.md for the measured quality/speed numbers that
run produced).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from onnxruntime.quantization import QuantType

from coro.recipes.canary_decoder_dynamic_quantization import ACCEPTED_SELECTOR, main


def _touching_quantize_dynamic(**kwargs):
    """Fake quantize_dynamic that creates an (empty) output file, like the real one."""
    Path(kwargs["model_output"]).write_bytes(b"")


class TestMain:
    def test_accepted_selector_is_quint8(self):
        """onnx-canary-split's decoder_quantization="dynamic_v1_quint8" selector
        must match ACCEPTED_SELECTOR exactly -- this is the contract the
        backend's docstring and settings.py's asr_decoder_quantization
        description both advertise.
        """
        assert ACCEPTED_SELECTOR == "dynamic_v1_quint8"

    def test_produces_both_quint8_and_qint8_variants(self, tmp_path):
        source = tmp_path / "decoder_step.onnx"
        source.write_bytes(b"")
        out_dir = tmp_path / "out"

        with patch(
            "coro.recipes.canary_decoder_dynamic_quantization.quantize_dynamic",
            autospec=True,
            side_effect=_touching_quantize_dynamic,
        ) as mock_quant:
            main(["--source", str(source), "--out-dir", str(out_dir)])

        assert (out_dir / "decoder_step.dynamic_v1_quint8.onnx").exists()
        assert (out_dir / "decoder_step.dynamic_v1_qint8.onnx").exists()
        assert mock_quant.call_count == 2
        weight_types = {c.kwargs["weight_type"] for c in mock_quant.call_args_list}
        assert weight_types == {QuantType.QUInt8, QuantType.QInt8}

    def test_only_quantizes_matmul(self, tmp_path):
        source = tmp_path / "decoder_step.onnx"
        source.write_bytes(b"")
        with patch(
            "coro.recipes.canary_decoder_dynamic_quantization.quantize_dynamic",
            autospec=True,
            side_effect=_touching_quantize_dynamic,
        ) as mock_quant:
            main(["--source", str(source), "--out-dir", str(tmp_path / "out")])

        assert mock_quant.call_count == 2
        assert all(
            c.kwargs["op_types_to_quantize"] == ["MatMul"] for c in mock_quant.call_args_list
        )

    def test_default_paths_follow_the_recipe_artifacts_convention(self, tmp_path):
        source_default = Path("recipe-artifacts") / "canary_split" / "decoder_step.onnx"
        with patch(
            "coro.recipes.canary_decoder_dynamic_quantization.quantize_dynamic",
            autospec=True,
            side_effect=_touching_quantize_dynamic,
        ) as mock_quant:
            main(["--out-dir", str(tmp_path / "out")])

        assert mock_quant.call_count == 2
        assert all(
            c.kwargs["model_input"] == str(source_default) for c in mock_quant.call_args_list
        )
