"""coro.recipes.canary_split_decoder: path resolution, extraction wiring, and
composition verification.

Heavy externals (``huggingface_hub``, ``onnx``, ``onnxruntime``, ``onnx_asr``)
are mocked -- these tests protect this recipe's own logic (naming/path
conventions, the composition-verification loop's pass/fail decision and
state advancement, argparse wiring), not onnx/onnxruntime's own behavior.
The real extraction is exercised end-to-end when the recipe is actually run
against a downloaded checkpoint (see the recipe's README.md).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coro.recipes.canary_split_decoder import (
    KV_TENSORS,
    SplitPaths,
    build,
    main,
    resolve_decoder_path,
    verify,
)


class _FakeGraph:
    def __init__(self, n_nodes: int = 0, initializer_dims: list[list[int]] | None = None):
        self.node = [None] * n_nodes
        self.initializer = [type("Init", (), {"dims": dims})() for dims in (initializer_dims or [])]


class TestResolveDecoderPath:
    def test_downloads_and_returns_decoder_model_path(self, tmp_path):
        with patch(
            "huggingface_hub.snapshot_download", autospec=True, return_value=str(tmp_path)
        ) as mock_dl:
            result = resolve_decoder_path()

        assert result == tmp_path / "decoder-model.onnx"
        mock_dl.assert_called_once_with(
            "istupakov/canary-1b-v2-onnx",
            allow_patterns=["decoder-model.onnx", "config.json", "vocab.txt"],
        )


class TestBuild:
    def test_extracts_the_16_kv_tensors_at_the_documented_cut_points(self, tmp_path):
        """The cut points are the recipe's actual contract -- not the shallower
        MatMul outputs. See the recipe's README.md for why.
        """
        src = tmp_path / "decoder-model.onnx"
        out_dir = tmp_path / "out"
        fake_model = MagicMock(graph=_FakeGraph())
        fake_graph_model = MagicMock(graph=_FakeGraph())

        with (
            patch(
                "onnx.load",
                autospec=True,
                side_effect=[fake_model, fake_graph_model, fake_graph_model],
            ),
            patch("onnx.shape_inference.infer_shapes", autospec=True, return_value=fake_model),
            patch("onnx.save", autospec=True),
            patch("onnx.utils.extract_model", autospec=True) as mock_extract,
        ):
            paths = build(src, out_dir)

        assert paths == SplitPaths(
            xattn_kv=out_dir / "xattn_kv.onnx", decoder_step=out_dir / "decoder_step.onnx"
        )
        assert mock_extract.call_count == 2
        xattn_call, decoder_call = mock_extract.call_args_list
        assert xattn_call.args[2] == ["encoder_embeddings"]
        assert xattn_call.args[3] == KV_TENSORS
        assert len(KV_TENSORS) == 16
        assert decoder_call.args[2] == ["input_ids", "encoder_mask", "decoder_mems", *KV_TENSORS]
        assert decoder_call.args[3] == ["logits", "decoder_hidden_states"]

    def test_creates_out_dir_and_cleans_up_the_intermediate_inferred_graph(self, tmp_path):
        src = tmp_path / "decoder-model.onnx"
        out_dir = tmp_path / "nested" / "out"
        fake = MagicMock(graph=_FakeGraph())

        with (
            patch("onnx.load", autospec=True, return_value=fake),
            patch("onnx.shape_inference.infer_shapes", autospec=True, return_value=fake),
            patch("onnx.save", autospec=True),
            patch("onnx.utils.extract_model", autospec=True),
        ):
            build(src, out_dir)

        assert out_dir.is_dir()
        assert not (out_dir / "decoder.inferred.onnx").exists()


class TestVerify:
    """The composition-verification loop's own logic: pass/fail decision and
    the input_ids/decoder_mems state it advances between steps.
    """

    def _sessions(self, *, fused_outputs, split_outputs):
        """Build fused/xattn_kv/decoder_step InferenceSession mocks.

        `fused_outputs`/`split_outputs` are lists of (logits, mems) pairs, one
        per decode step.
        """
        fused = MagicMock()
        fused.run.side_effect = [list(pair) for pair in fused_outputs]
        xattn_kv = MagicMock()
        xattn_kv.run.return_value = [np.zeros((1, 1, 1)) for _ in KV_TENSORS]
        decoder_step = MagicMock()
        decoder_step.run.side_effect = [list(pair) for pair in split_outputs]
        return fused, xattn_kv, decoder_step

    def test_matching_outputs_across_all_steps_is_ok(self, tmp_path):
        logits = np.ones((1, 1, 5), dtype=np.float32)
        mems = np.zeros((10, 1, 1, 1024), dtype=np.float32)
        fused, xattn_kv, decoder_step = self._sessions(
            fused_outputs=[(logits, mems)] * 2, split_outputs=[(logits, mems)] * 2
        )
        paths = SplitPaths(xattn_kv=Path("xattn"), decoder_step=Path("step"))

        with (
            patch(
                "onnxruntime.InferenceSession",
                autospec=True,
                side_effect=[fused, xattn_kv, decoder_step],
            ),
            patch(
                "coro.recipes.canary_split_decoder._real_encoder_embeddings",
                autospec=True,
                return_value=(np.zeros((1, 4, 1024), dtype=np.float32), np.ones((1, 4))),
            ),
        ):
            ok = verify(Path("fused.onnx"), paths, Path("holdout.wav"), n_steps=2)

        assert ok is True

    def test_a_mismatched_step_makes_the_whole_verification_fail(self, tmp_path):
        """One divergent step must fail the whole run, not just be logged."""
        logits = np.ones((1, 1, 5), dtype=np.float32)
        mems = np.zeros((10, 1, 1, 1024), dtype=np.float32)
        divergent_logits = logits + 1.0
        fused, xattn_kv, decoder_step = self._sessions(
            fused_outputs=[(logits, mems), (logits, mems)],
            split_outputs=[(logits, mems), (divergent_logits, mems)],
        )

        paths = SplitPaths(xattn_kv=Path("xattn"), decoder_step=Path("step"))
        with (
            patch(
                "onnxruntime.InferenceSession",
                autospec=True,
                side_effect=[fused, xattn_kv, decoder_step],
            ),
            patch(
                "coro.recipes.canary_split_decoder._real_encoder_embeddings",
                autospec=True,
                return_value=(np.zeros((1, 4, 1024), dtype=np.float32), np.ones((1, 4))),
            ),
        ):
            ok = verify(Path("fused.onnx"), paths, Path("holdout.wav"), n_steps=2)

        assert ok is False

    def test_first_step_feeds_the_full_prefix_later_steps_feed_only_the_new_token(self, tmp_path):
        """decoder_mems.shape[2] == 0 (step 0) must feed the full input_ids
        prefix; every later step (mems already populated) must feed only the
        newest token -- this is the real ``_decoding`` loop's own contract.
        """
        logits = np.ones((1, 1, 5), dtype=np.float32)
        # verify() itself starts decoder_mems empty (shape[2]==0); returning
        # this non-empty shape from step 0's fused output is what makes step
        # 1's decoder_mems non-empty, exercising the "only newest token" branch.
        mems_after_step = np.zeros((10, 1, 1, 1024), dtype=np.float32)
        fused, xattn_kv, decoder_step = self._sessions(
            fused_outputs=[(logits, mems_after_step), (logits, mems_after_step)],
            split_outputs=[(logits, mems_after_step), (logits, mems_after_step)],
        )

        paths = SplitPaths(xattn_kv=Path("xattn"), decoder_step=Path("step"))
        with (
            patch(
                "onnxruntime.InferenceSession",
                autospec=True,
                side_effect=[fused, xattn_kv, decoder_step],
            ),
            patch(
                "coro.recipes.canary_split_decoder._real_encoder_embeddings",
                autospec=True,
                return_value=(np.zeros((1, 4, 1024), dtype=np.float32), np.ones((1, 4))),
            ),
        ):
            verify(Path("fused.onnx"), paths, Path("holdout.wav"), n_steps=2)

        first_call_feed = fused.run.call_args_list[0].args[1]
        second_call_feed = fused.run.call_args_list[1].args[1]
        assert first_call_feed["input_ids"].shape[1] == 3  # full 3-token prefix
        assert second_call_feed["input_ids"].shape[1] == 1  # only the newest token


class TestMain:
    def test_wires_resolve_build_and_verify_together(self, tmp_path):
        fake_paths = SplitPaths(
            xattn_kv=tmp_path / "xattn_kv.onnx", decoder_step=tmp_path / "decoder_step.onnx"
        )
        with (
            patch(
                "coro.recipes.canary_split_decoder.resolve_decoder_path",
                autospec=True,
                return_value=tmp_path / "decoder-model.onnx",
            ) as mock_resolve,
            patch(
                "coro.recipes.canary_split_decoder.build", autospec=True, return_value=fake_paths
            ) as mock_build,
            patch(
                "coro.recipes.canary_split_decoder.verify", autospec=True, return_value=True
            ) as mock_verify,
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["--out-dir", str(tmp_path / "out"), "--verify-steps", "7"])

        assert exc_info.value.code == 0
        mock_resolve.assert_called_once_with()
        mock_build.assert_called_once_with(tmp_path / "decoder-model.onnx", tmp_path / "out")
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs["n_steps"] == 7

    def test_exits_nonzero_when_verification_fails(self, tmp_path):
        with (
            patch(
                "coro.recipes.canary_split_decoder.resolve_decoder_path",
                autospec=True,
                return_value=tmp_path / "decoder-model.onnx",
            ),
            patch(
                "coro.recipes.canary_split_decoder.build",
                autospec=True,
                return_value=SplitPaths(xattn_kv=tmp_path / "a", decoder_step=tmp_path / "b"),
            ),
            patch("coro.recipes.canary_split_decoder.verify", autospec=True, return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["--out-dir", str(tmp_path)])

        assert exc_info.value.code == 1

    def test_out_dir_defaults_under_recipe_artifacts(self, tmp_path):
        with (
            patch(
                "coro.recipes.canary_split_decoder.resolve_decoder_path",
                autospec=True,
                return_value=tmp_path / "decoder-model.onnx",
            ),
            patch(
                "coro.recipes.canary_split_decoder.build",
                autospec=True,
                return_value=SplitPaths(xattn_kv=tmp_path / "a", decoder_step=tmp_path / "b"),
            ) as mock_build,
            patch("coro.recipes.canary_split_decoder.verify", autospec=True, return_value=True),
            pytest.raises(SystemExit),
        ):
            main([])

        out_dir = mock_build.call_args.args[1]
        assert out_dir == Path("recipe-artifacts") / "canary_split"
