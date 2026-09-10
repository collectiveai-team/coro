"""onnx-canary-split backend: Hugging Face artifact resolution.

``model_asr`` may be a local directory or a HF repo id; these tests pin the
hub half of that contract (``tests/recipes/test_canary_split_decoder.py``'s
convention: mock ``huggingface_hub.snapshot_download``, never the code under
test, and never the network):

- the default INT8/INT8 selection downloads exactly the needed files -- never
  the 609 MB fp32 ``decoder_step.onnx``, never an unselected encoder;
- the fp32-encoder selector (``"fp32"`` and, for this backend only, ``None``)
  pulls the encoder from ``istupakov/canary-1b-v2-onnx`` and everything else
  from the split repo, with ``model_files`` assembled across both snapshots;
- a local directory makes no hub call at all;
- a file still missing after download raises the backend's FileNotFoundError
  shape with the repo id and filename in the message;
- ``hf_token`` is forwarded to every download.

``onnxruntime.InferenceSession`` is stubbed exactly as in
``test_onnx_canary_split_backend.py`` (no real ``.onnx`` files here).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from coro.backends.asr.onnx_canary_split import build_onnx_canary_split_adapter

_SPLIT_REPO = "collectiveai/canary-1b-v2-onnx-split-int8"
_FP32_ENCODER_REPO = "istupakov/canary-1b-v2-onnx"

_INT8_INT8_PATTERNS = [
    "encoder-model.static_qdq_v4_pct_excl.onnx",
    "encoder-model.static_qdq_v4_pct_excl.onnx.data",
    "decoder_step.dynamic_v1_quint8.onnx",
    "xattn_kv.onnx",
    "vocab.txt",
    "config.json",
]


def _write_vocab(path, pieces, blank_id):
    with path.open("w", encoding="utf-8") as f:
        for idx, piece in enumerate(pieces):
            f.write(f"{piece} {idx}\n")
        f.write(f"<blk> {blank_id}\n")


_REQUIRED_TOKENS = [
    "<unk>",
    "\u2581",  # SentencePiece space marker; onnx_asr's vocab loader maps it to " "
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
    "<|en|>",
    "<|pnc|>",
    "<|noitn|>",
    "<|notimestamp|>",
    "<|nodiarize|>",
    "<|endoftext|>",
]


def _populate_split_snapshot(directory):
    """Write the collectiveai split repo's INT8/INT8 snapshot contents."""
    (directory / "encoder-model.static_qdq_v4_pct_excl.onnx").write_bytes(b"")
    (directory / "encoder-model.static_qdq_v4_pct_excl.onnx.data").write_bytes(b"")
    (directory / "decoder_step.dynamic_v1_quint8.onnx").write_bytes(b"")
    (directory / "xattn_kv.onnx").write_bytes(b"")
    _write_vocab(directory / "vocab.txt", _REQUIRED_TOKENS, blank_id=len(_REQUIRED_TOKENS))
    return directory


class TestInt8Int8HubResolution:
    def test_downloads_exactly_the_needed_files(self, tmp_path):
        """One snapshot_download, exact allow_patterns; the fp32 decoder is never pulled."""
        _populate_split_snapshot(tmp_path)
        with (
            patch(
                "huggingface_hub.snapshot_download", autospec=True, return_value=str(tmp_path)
            ) as mock_dl,
            patch("onnxruntime.InferenceSession", autospec=True, return_value=MagicMock()),
        ):
            adapter = build_onnx_canary_split_adapter(
                _SPLIT_REPO,
                device="cpu",
                quantization="static_qdq_v4_pct_excl",
                decoder_quantization="dynamic_v1_quint8",
            )

        assert adapter is not None
        mock_dl.assert_called_once_with(_SPLIT_REPO, allow_patterns=_INT8_INT8_PATTERNS, token=None)
        patterns = mock_dl.call_args.kwargs["allow_patterns"]
        assert "decoder_step.onnx" not in patterns  # the 609 MB fp32 decoder
        assert "encoder-model.onnx" not in patterns  # the fp32 encoder

    def test_forwards_hf_token(self, tmp_path):
        expected = "tok"
        _populate_split_snapshot(tmp_path)
        with (
            patch(
                "huggingface_hub.snapshot_download", autospec=True, return_value=str(tmp_path)
            ) as mock_dl,
            patch("onnxruntime.InferenceSession", autospec=True, return_value=MagicMock()),
        ):
            build_onnx_canary_split_adapter(
                _SPLIT_REPO,
                device="cpu",
                quantization="static_qdq_v4_pct_excl",
                decoder_quantization="dynamic_v1_quint8",
                hf_token=expected,
            )

        assert mock_dl.call_args.kwargs["token"] == expected


class TestFp32EncoderTwoRepoResolution:
    @pytest.mark.parametrize("quantization", ["fp32", None])
    def test_encoder_from_upstream_and_everything_else_from_the_split_repo(
        self, tmp_path, quantization
    ):
        """Resolve the fp32 encoder from a second repo for both fp32 selectors."""
        upstream = tmp_path / "upstream"
        split = tmp_path / "split"
        upstream.mkdir()
        split.mkdir()
        _populate_split_snapshot(split)
        (upstream / "encoder-model.onnx").write_bytes(b"")
        (upstream / "encoder-model.onnx.data").write_bytes(b"")
        (split / "decoder_step.onnx").write_bytes(b"")  # fp32 decoder lives in the split repo

        snapshots = {_FP32_ENCODER_REPO: str(upstream), _SPLIT_REPO: str(split)}
        session_paths: list[str] = []

        def _fake_session(path, **_kwargs):
            session_paths.append(str(path))
            return MagicMock()

        def _fake_download(repo_id, **_kwargs):
            return snapshots[repo_id]

        with (
            patch(
                "huggingface_hub.snapshot_download", autospec=True, side_effect=_fake_download
            ) as mock_dl,
            patch("onnxruntime.InferenceSession", autospec=True, side_effect=_fake_session),
        ):
            build_onnx_canary_split_adapter(_SPLIT_REPO, device="cpu", quantization=quantization)

        repo_calls = [
            (call.args[0], tuple(call.kwargs["allow_patterns"])) for call in mock_dl.call_args_list
        ]
        assert repo_calls == [
            (_FP32_ENCODER_REPO, ("encoder-model.onnx", "encoder-model.onnx.data")),
            (
                _SPLIT_REPO,
                ("decoder_step.onnx", "xattn_kv.onnx", "vocab.txt", "config.json"),
            ),
        ]
        # model_files assembled across the two snapshots, not one directory.
        assert any(str(upstream / "encoder-model.onnx") == p for p in session_paths)
        assert any(str(split / "decoder_step.onnx") == p for p in session_paths)
        assert not any(
            p.endswith("encoder-model.static_qdq_v4_pct_excl.onnx") for p in session_paths
        )


class TestLocalDirectoryBypassesTheHub:
    def test_no_download_call(self, tmp_path):
        """An existing local directory resolves entirely offline."""
        (tmp_path / "encoder-model.onnx").write_bytes(b"")
        (tmp_path / "xattn_kv.onnx").write_bytes(b"")
        (tmp_path / "decoder_step.onnx").write_bytes(b"")
        _write_vocab(tmp_path / "vocab.txt", _REQUIRED_TOKENS, blank_id=len(_REQUIRED_TOKENS))

        with (
            patch("huggingface_hub.snapshot_download", autospec=True) as mock_dl,
            patch("onnxruntime.InferenceSession", autospec=True, return_value=MagicMock()),
        ):
            adapter = build_onnx_canary_split_adapter(str(tmp_path), device="cpu")

        assert adapter is not None
        mock_dl.assert_not_called()


class TestMissingFileAfterDownload:
    def test_raises_file_not_found_with_repo_id_and_filename(self, tmp_path):
        """The post-download check keeps the local FileNotFoundError shape plus repo context."""
        _populate_split_snapshot(tmp_path)
        (tmp_path / "encoder-model.static_qdq_v4_pct_excl.onnx").unlink()

        with (
            patch("huggingface_hub.snapshot_download", autospec=True, return_value=str(tmp_path)),
            patch("onnxruntime.InferenceSession", autospec=True, return_value=MagicMock()),
            pytest.raises(FileNotFoundError) as excinfo,
        ):
            build_onnx_canary_split_adapter(
                _SPLIT_REPO,
                device="cpu",
                quantization="static_qdq_v4_pct_excl",
                decoder_quantization="dynamic_v1_quint8",
            )

        message = str(excinfo.value)
        assert "Missing required onnx-canary-split artifact" in message
        assert _SPLIT_REPO in message
        assert "encoder-model.static_qdq_v4_pct_excl.onnx" in message
