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


def _run_recipe(tmp_path):
    """Run the recipe on an empty checkpoint; return the output dir and the patched loader."""
    checkpoint = tmp_path / "checkpoint.nemo"
    checkpoint.write_bytes(b"")
    out_dir = tmp_path / "out"
    with patch(
        "nemo.collections.asr.models.ASRModel.restore_from",
        autospec=True,
        return_value=_fake_model(),
    ) as mock_restore:
        main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])
    return out_dir, mock_restore


class TestMain:
    def test_restores_the_given_checkpoint(self, tmp_path):
        _, mock_restore = _run_recipe(tmp_path)

        mock_restore.assert_called_once()
        assert mock_restore.call_args.args[0] == str(tmp_path / "checkpoint.nemo")

    def test_writes_every_prompt_kernel_weight(self, tmp_path):
        out_dir, _ = _run_recipe(tmp_path)

        weights = np.load(out_dir / "prompt_kernel_cache.npz")
        assert set(weights.files) == set(_fake_model().prompt_kernel.state_dict())

    def test_writes_the_tokenizer_metadata(self, tmp_path):
        out_dir, _ = _run_recipe(tmp_path)

        meta = json.loads((out_dir / "prompt_kernel_cache.json").read_text())
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
