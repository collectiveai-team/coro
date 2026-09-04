"""coro.recipes.parakeet_prompt_encoder_static_qdq: calibration wiring and the
quantize_static parameters the encoded_lengths corruption bug requires.

``onnxruntime.quantization``'s heavy calls (``quant_pre_process``,
``quantize_static``) are mocked -- these tests protect this recipe's own
contract: the calibration manifest error message, the preprocessing-cache
skip, and (most importantly) that the specific `op_types_to_quantize`/
`per_channel`/`reduce_range` parameters the README.md documents as fixing a
real diagnosed bug are never silently dropped.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnxruntime.quantization import QuantType

from coro.recipes.parakeet_prompt_encoder_static_qdq import (
    ACCEPTED_SELECTOR,
    ParakeetEncoderCalibrationReader,
    _load_calibration_wavs,
    main,
)


class TestLoadCalibrationWavs:
    def test_raises_a_helpful_error_when_the_manifest_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="calibration_corpus"):
            _load_calibration_wavs(tmp_path / "does_not_exist.json")

    def test_returns_the_path_field_of_every_manifest_entry(self, tmp_path):
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {"path": "es-1.wav", "lang": "es", "duration_s": 5.0, "source": "x"},
                    {"path": "en-1.wav", "lang": "en", "duration_s": 6.0, "source": "y"},
                ]
            )
        )

        assert _load_calibration_wavs(manifest_path) == ["es-1.wav", "en-1.wav"]


class TestParakeetEncoderCalibrationReader:
    def _fake_manager(self):
        manager = MagicMock()
        manager._create_preprocessor.return_value = lambda waveforms, lens: (
            np.zeros((1, 80, 10), dtype=np.float32),
            lens,
        )
        return manager

    def test_yields_one_dict_per_clip_then_none(self, tmp_path):
        import soundfile as sf

        wav_path = tmp_path / "clip.wav"
        sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)

        with patch("onnx_asr.loader.Manager", autospec=True, return_value=self._fake_manager()):
            reader = ParakeetEncoderCalibrationReader([str(wav_path), str(wav_path)])

        first = reader.get_next()
        second = reader.get_next()
        third = reader.get_next()

        assert first is not None
        assert set(first) == {"audio_signal", "length"}
        assert second is not None
        assert third is None

    def test_rejects_a_clip_at_the_wrong_sample_rate(self, tmp_path):
        import soundfile as sf

        wav_path = tmp_path / "clip.wav"
        sf.write(str(wav_path), np.zeros(8000, dtype=np.float32), 8000)

        with (
            patch("onnx_asr.loader.Manager", autospec=True, return_value=self._fake_manager()),
            pytest.raises(ValueError, match="8000 Hz"),
        ):
            ParakeetEncoderCalibrationReader([str(wav_path)])


class TestMain:
    def _run(self, tmp_path, *, preexisting_preprocessed: bool = False):
        source = tmp_path / "encoder-encoder.onnx"
        source.write_bytes(b"")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest_path.write_text(json.dumps([{"path": "clip.wav"}]))

        if preexisting_preprocessed:
            (out_dir / "encoder-encoder.preprocessed.onnx").write_bytes(b"")

        with (
            patch(
                "coro.recipes.parakeet_prompt_encoder_static_qdq.quant_pre_process",
                autospec=True,
            ) as mock_preprocess,
            patch(
                "coro.recipes.parakeet_prompt_encoder_static_qdq.quantize_static",
                autospec=True,
            ) as mock_quantize,
            patch(
                "coro.recipes.parakeet_prompt_encoder_static_qdq.ParakeetEncoderCalibrationReader",
                autospec=True,
            ) as mock_reader_cls,
        ):
            main(
                [
                    "--source",
                    str(source),
                    "--calibration-manifest",
                    str(manifest_path),
                    "--out-dir",
                    str(out_dir),
                ]
            )
        return out_dir, mock_preprocess, mock_quantize, mock_reader_cls

    def test_output_filename_uses_the_accepted_selector(self, tmp_path):
        out_dir, _, mock_quantize, _ = self._run(tmp_path)

        assert ACCEPTED_SELECTOR == "static_qdq_v3"
        _, kwargs = mock_quantize.call_args
        assert kwargs["model_output"] == str(out_dir / "encoder-encoder.static_qdq_v3.onnx")

    def test_quantize_static_parameters_match_the_diagnosed_fix(self, tmp_path):
        """These exact parameters fix a real, diagnosed bug (see README.md) --
        this test exists so nobody accidentally "simplifies" them away.
        """
        _, _, mock_quantize, _ = self._run(tmp_path)

        _, kwargs = mock_quantize.call_args
        assert kwargs["activation_type"] == QuantType.QUInt8
        assert kwargs["weight_type"] == QuantType.QInt8
        assert kwargs["reduce_range"] is True
        assert kwargs["per_channel"] is True
        assert kwargs["op_types_to_quantize"] == ["Conv", "MatMul", "Gemm"]

    def test_preprocessing_is_skipped_when_the_cache_already_exists(self, tmp_path):
        _, mock_preprocess, _, _ = self._run(tmp_path, preexisting_preprocessed=True)

        mock_preprocess.assert_not_called()

    def test_preprocessing_runs_when_no_cache_exists(self, tmp_path):
        out_dir, mock_preprocess, _, _ = self._run(tmp_path, preexisting_preprocessed=False)

        mock_preprocess.assert_called_once()
        args = mock_preprocess.call_args.args
        assert args[1] == str(out_dir / "encoder-encoder.preprocessed.onnx")

    def test_calibration_reader_is_built_from_the_manifest(self, tmp_path):
        _, _, _, mock_reader_cls = self._run(tmp_path)

        mock_reader_cls.assert_called_once_with(["clip.wav"])

    def test_missing_calibration_manifest_raises_before_quantizing(self, tmp_path):
        source = tmp_path / "encoder-encoder.onnx"
        source.write_bytes(b"")

        with (
            patch(
                "coro.recipes.parakeet_prompt_encoder_static_qdq.quant_pre_process",
                autospec=True,
            ),
            patch(
                "coro.recipes.parakeet_prompt_encoder_static_qdq.quantize_static",
                autospec=True,
            ) as mock_quantize,
            pytest.raises(FileNotFoundError),
        ):
            main(
                [
                    "--source",
                    str(source),
                    "--calibration-manifest",
                    str(tmp_path / "missing.json"),
                    "--out-dir",
                    str(tmp_path / "out"),
                ]
            )

        mock_quantize.assert_not_called()
