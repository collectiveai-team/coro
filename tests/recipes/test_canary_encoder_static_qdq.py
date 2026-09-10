"""coro.recipes.canary_encoder_static_qdq: calibration wiring and the
quantize_static parameters the rejected first attempt got wrong.

``onnxruntime.quantization``'s heavy calls (``quant_pre_process``,
``quantize_static``) are mocked -- these tests protect this recipe's own
contract: manifest loading (including the relocation fallback), the chunked
calibration-reader protocol ORT requires for histogram methods, and that the
percentile calibration method and measured `nodes_to_exclude` list -- the two
things that turned a rejected result into an accepted one -- are never
silently dropped.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnxruntime.quantization import CalibrationMethod, QuantType

from coro.recipes.canary_encoder_static_qdq import (
    ACCEPTED_SELECTOR,
    CALIBRATION_PERCENTILE,
    CALIBRATION_STRIDE,
    NODES_TO_EXCLUDE,
    CanaryEncoderCalibrationReader,
    _load_calibration_wavs,
    main,
)


class TestLoadCalibrationWavs:
    def test_raises_a_helpful_error_when_the_manifest_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="calibration_corpus"):
            _load_calibration_wavs(tmp_path / "does_not_exist.json")

    def test_returns_recorded_paths_when_they_still_resolve(self, tmp_path):
        clip = tmp_path / "es-1.wav"
        clip.write_bytes(b"")
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest_path.write_text(json.dumps([{"path": str(clip), "lang": "es"}]))

        assert _load_calibration_wavs(manifest_path) == [str(clip)]

    def test_falls_back_to_the_corpus_layout_next_to_the_manifest(self, tmp_path):
        """A corpus moved between worktrees keeps its layout but not its
        recorded absolute paths -- that must not force a re-download.
        """
        (tmp_path / "es").mkdir()
        relocated = tmp_path / "es" / "es-1.wav"
        relocated.write_bytes(b"")
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest_path.write_text(json.dumps([{"path": "/gone/es/es-1.wav", "lang": "es"}]))

        assert _load_calibration_wavs(manifest_path) == [str(relocated)]

    def test_raises_when_a_clip_resolves_nowhere(self, tmp_path):
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest_path.write_text(json.dumps([{"path": "/gone/es/es-1.wav", "lang": "es"}]))

        with pytest.raises(FileNotFoundError, match="missing at both"):
            _load_calibration_wavs(manifest_path)


class TestCanaryEncoderCalibrationReader:
    def _fake_manager(self):
        manager = MagicMock()
        manager._create_preprocessor.return_value = lambda waveforms, lens: (
            np.zeros((1, 128, 10), dtype=np.float32),
            lens,
        )
        return manager

    def _reader(self, tmp_path, count: int) -> CanaryEncoderCalibrationReader:
        import soundfile as sf

        wav_path = tmp_path / "clip.wav"
        sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), 16000)
        with patch("onnx_asr.loader.Manager", autospec=True, return_value=self._fake_manager()):
            return CanaryEncoderCalibrationReader([str(wav_path)] * count)

    def test_yields_one_dict_per_clip_then_none(self, tmp_path):
        reader = self._reader(tmp_path, 2)

        first = reader.get_next()
        second = reader.get_next()

        assert first is not None
        assert set(first) == {"audio_signal", "length"}
        assert second is not None
        assert reader.get_next() is None

    def test_rejects_a_clip_at_the_wrong_sample_rate(self, tmp_path):
        import soundfile as sf

        wav_path = tmp_path / "clip.wav"
        sf.write(str(wav_path), np.zeros(8000, dtype=np.float32), 8000)

        with (
            patch("onnx_asr.loader.Manager", autospec=True, return_value=self._fake_manager()),
            pytest.raises(ValueError, match="8000 Hz"),
        ):
            CanaryEncoderCalibrationReader([str(wav_path)])

    def test_reports_its_length_so_ort_can_chunk_calibration(self, tmp_path):
        assert len(self._reader(tmp_path, 3)) == 3

    def test_set_range_restricts_iteration_to_one_chunk(self, tmp_path):
        """ORT's chunked calibration path calls set_range then drains the
        reader once per chunk; without this the histogram collector sees the
        whole corpus at once and exhausts memory.
        """
        reader = self._reader(tmp_path, 4)

        reader.set_range(start_index=1, end_index=3)
        drained = [reader.get_next(), reader.get_next(), reader.get_next()]

        assert drained[0] is not None
        assert drained[1] is not None
        assert drained[2] is None

    def test_rewind_returns_to_the_start_of_the_current_chunk(self, tmp_path):
        reader = self._reader(tmp_path, 4)
        reader.set_range(start_index=2, end_index=4)
        reader.get_next()
        reader.get_next()

        reader.rewind()

        assert reader.get_next() is not None


class TestMain:
    def _run(self, tmp_path, *, preexisting_preprocessed: bool = False):
        source = tmp_path / "encoder-model.onnx"
        source.write_bytes(b"")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"")
        manifest_path = tmp_path / "calibration_manifest.json"
        manifest_path.write_text(json.dumps([{"path": str(clip), "lang": "es"}]))

        if preexisting_preprocessed:
            (out_dir / "encoder-model.preprocessed.onnx").write_bytes(b"")

        with (
            patch(
                "coro.recipes.canary_encoder_static_qdq.quant_pre_process",
                autospec=True,
            ) as mock_preprocess,
            patch(
                "coro.recipes.canary_encoder_static_qdq.quantize_static",
                autospec=True,
            ) as mock_quantize,
            patch(
                "coro.recipes.canary_encoder_static_qdq.CanaryEncoderCalibrationReader",
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
        return out_dir, mock_preprocess, mock_quantize, mock_reader_cls, str(clip)

    def test_output_filename_uses_the_accepted_selector(self, tmp_path):
        out_dir, _, mock_quantize, _, _ = self._run(tmp_path)

        assert ACCEPTED_SELECTOR == "static_qdq_v4_pct_excl"
        _, kwargs = mock_quantize.call_args
        assert kwargs["model_output"] == str(out_dir / "encoder-model.static_qdq_v4_pct_excl.onnx")

    def test_calibration_is_percentile_not_minmax(self, tmp_path):
        """MinMax calibration is exactly what the rejected v3 attempt used --
        this test exists so nobody reverts to ORT's default.
        """
        _, _, mock_quantize, _, _ = self._run(tmp_path)

        _, kwargs = mock_quantize.call_args
        assert kwargs["calibrate_method"] == CalibrationMethod.Percentile
        assert kwargs["extra_options"]["CalibPercentile"] == CALIBRATION_PERCENTILE

    def test_calibration_is_chunked_one_clip_at_a_time(self, tmp_path):
        """ORT's HistogramCollector calls np.asarray() over a whole chunk,
        which raises on this corpus's variable-length clips at stride > 1.
        """
        _, _, mock_quantize, _, _ = self._run(tmp_path)

        _, kwargs = mock_quantize.call_args
        assert CALIBRATION_STRIDE == 1
        assert kwargs["extra_options"]["CalibStridedMinMax"] == 1

    def test_the_measured_sensitivity_exclusions_are_passed_through(self, tmp_path):
        _, _, mock_quantize, _, _ = self._run(tmp_path)

        _, kwargs = mock_quantize.call_args
        assert kwargs["nodes_to_exclude"] == NODES_TO_EXCLUDE
        assert len(NODES_TO_EXCLUDE) == 32

    def test_quantize_static_parameters_match_the_diagnosed_fix(self, tmp_path):
        _, _, mock_quantize, _, _ = self._run(tmp_path)

        _, kwargs = mock_quantize.call_args
        assert kwargs["activation_type"] == QuantType.QUInt8
        assert kwargs["weight_type"] == QuantType.QInt8
        assert kwargs["reduce_range"] is True
        assert kwargs["per_channel"] is True
        assert kwargs["op_types_to_quantize"] == ["Conv", "MatMul", "Gemm"]

    def test_activation_ranges_are_cached_next_to_the_output(self, tmp_path):
        out_dir, _, mock_quantize, _, _ = self._run(tmp_path)

        _, kwargs = mock_quantize.call_args
        assert kwargs["calibration_cache_path"] == str(
            out_dir / "encoder-model.calibration_percentile.json"
        )

    def test_preprocessing_is_skipped_when_the_cache_already_exists(self, tmp_path):
        _, mock_preprocess, _, _, _ = self._run(tmp_path, preexisting_preprocessed=True)

        mock_preprocess.assert_not_called()

    def test_preprocessing_runs_when_no_cache_exists(self, tmp_path):
        out_dir, mock_preprocess, _, _, _ = self._run(tmp_path, preexisting_preprocessed=False)

        mock_preprocess.assert_called_once()
        args = mock_preprocess.call_args.args
        assert args[1] == str(out_dir / "encoder-model.preprocessed.onnx")

    def test_calibration_reader_is_built_from_the_manifest(self, tmp_path):
        _, _, _, mock_reader_cls, clip = self._run(tmp_path)

        mock_reader_cls.assert_called_once_with([clip])

    def test_missing_calibration_manifest_raises_before_quantizing(self, tmp_path):
        source = tmp_path / "encoder-model.onnx"
        source.write_bytes(b"")

        with (
            patch(
                "coro.recipes.canary_encoder_static_qdq.quant_pre_process",
                autospec=True,
            ),
            patch(
                "coro.recipes.canary_encoder_static_qdq.quantize_static",
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
