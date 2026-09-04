"""coro.recipes.parakeet_prompt_kernel_cache: checkpoint extraction and output contract.

``nemo.collections.asr.models.ASRModel.restore_from`` is patched with a fake
model exposing exactly the surface this recipe reads (``tokenizer``,
``joint``, ``cfg.model_defaults``, ``prompt_kernel``) -- loading a real ~4 GB
checkpoint is out of scope for a unit test; this recipe's own contract
(what it extracts, and the .npz/.json shape it writes) is what these tests
protect.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from coro.recipes.parakeet_prompt_kernel_cache import main


def _fake_model(*, blank_plus_one: int = 1025, vocab_size: int = 1024):
    prompt_kernel = torch.nn.Sequential(
        torch.nn.Linear(1026, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 1024)
    )
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=vocab_size),
        joint=SimpleNamespace(num_classes_with_blank=blank_plus_one),
        cfg=SimpleNamespace(model_defaults={"prompt_dictionary": {"es": 0, "en": 1, "pt": 2}}),
        prompt_kernel=prompt_kernel,
    )
    model.eval = lambda: model
    return model


class TestMain:
    def test_writes_prompt_kernel_weights_and_metadata(self, tmp_path):
        checkpoint = tmp_path / "checkpoint.nemo"
        checkpoint.write_bytes(b"")
        out_dir = tmp_path / "out"

        with patch(
            "nemo.collections.asr.models.ASRModel.restore_from",
            autospec=True,
            return_value=_fake_model(),
        ) as mock_restore:
            main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])

        mock_restore.assert_called_once()
        assert mock_restore.call_args.args[0] == str(checkpoint)

        cache_npz = out_dir / "prompt_kernel_cache.npz"
        cache_json = out_dir / "prompt_kernel_cache.json"
        assert cache_npz.exists()
        assert cache_json.exists()

        weights = np.load(cache_npz)
        expected_keys = {name for name, _ in _fake_model().prompt_kernel.state_dict().items()}
        assert set(weights.files) == expected_keys

        meta = json.loads(cache_json.read_text())
        assert meta == {
            "vocab_size": 1024,
            "blank_id": 1024,
            "prompt_dictionary": {"es": 0, "en": 1, "pt": 2},
        }

    def test_blank_id_is_num_classes_with_blank_minus_one(self, tmp_path):
        checkpoint = tmp_path / "checkpoint.nemo"
        checkpoint.write_bytes(b"")
        out_dir = tmp_path / "out"

        with patch(
            "nemo.collections.asr.models.ASRModel.restore_from",
            autospec=True,
            return_value=_fake_model(blank_plus_one=42),
        ):
            main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])

        meta = json.loads((out_dir / "prompt_kernel_cache.json").read_text())
        assert meta["blank_id"] == 41

    def test_missing_checkpoint_argument_is_required(self):
        with pytest.raises(SystemExit):
            main([])

    def test_default_out_dir_follows_the_recipe_artifacts_convention(self, tmp_path, monkeypatch):
        # RECIPE_ARTIFACTS_DIR is a relative path resolved against cwd -- chdir
        # into an isolated tmp_path so exercising the real default writes
        # nothing outside this test's own sandbox.
        monkeypatch.chdir(tmp_path)
        checkpoint = tmp_path / "checkpoint.nemo"
        checkpoint.write_bytes(b"")

        with patch(
            "nemo.collections.asr.models.ASRModel.restore_from",
            autospec=True,
            return_value=_fake_model(),
        ):
            main(["--checkpoint", str(checkpoint)])

        from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

        assert (
            tmp_path / RECIPE_ARTIFACTS_DIR / "parakeet_prompt" / "prompt_kernel_cache.npz"
        ).exists()
