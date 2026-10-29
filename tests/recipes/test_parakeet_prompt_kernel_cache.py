"""coro.recipes.parakeet_prompt_kernel_cache: checkpoint extraction and output contract.

NeMo's ``ASRModel.restore_from`` is a third-party loader with no port of the
project around it, so it is replaced by an in-memory checkpoint store holding
fake models that expose exactly the surface this recipe reads (``tokenizer``,
``joint``, ``cfg.model_defaults``, ``prompt_kernel``) -- loading a real ~4 GB
checkpoint is out of scope for a unit test; this recipe's own contract (what
it extracts, and the .npz/.json shape it writes) is what these tests protect.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nemo.collections.asr.models import ASRModel

from coro.recipes.parakeet_prompt_kernel_cache import main
from coro.recipes.paths import RECIPE_ARTIFACTS_DIR


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


class _InMemoryCheckpoints:
    """Stands in for NeMo's checkpoint loader: restores only the checkpoints it holds."""

    def __init__(self) -> None:
        self._models: dict[str, SimpleNamespace] = {}

    def add(self, path, model) -> None:
        path.write_bytes(b"")
        self._models[str(path)] = model

    def restore_from(self, restore_path: str, map_location=None):
        return self._models[restore_path]


@pytest.fixture
def checkpoints(monkeypatch):
    store = _InMemoryCheckpoints()
    monkeypatch.setattr(ASRModel, "restore_from", store.restore_from)
    return store


def _run_recipe(tmp_path, checkpoints, model=None):
    """Run the recipe on a checkpoint holding ``model``; return the output dir."""
    checkpoint = tmp_path / "checkpoint.nemo"
    checkpoints.add(checkpoint, model or _fake_model())
    out_dir = tmp_path / "out"
    main(["--checkpoint", str(checkpoint), "--out-dir", str(out_dir)])
    return out_dir


def _meta(out_dir):
    return json.loads((out_dir / "prompt_kernel_cache.json").read_text())


class TestMain:
    def test_extracts_from_the_given_checkpoint(self, tmp_path, checkpoints):
        checkpoints.add(tmp_path / "other.nemo", _fake_model(vocab_size=7))
        out_dir = _run_recipe(tmp_path, checkpoints, _fake_model(vocab_size=512))

        assert _meta(out_dir)["vocab_size"] == 512

    def test_writes_every_prompt_kernel_weight(self, tmp_path, checkpoints):
        model = _fake_model()
        out_dir = _run_recipe(tmp_path, checkpoints, model)

        weights = np.load(out_dir / "prompt_kernel_cache.npz")
        state = model.prompt_kernel.state_dict()
        assert set(weights.files) == set(state)
        assert all(np.array_equal(weights[name], state[name].numpy()) for name in state)

    def test_writes_the_tokenizer_metadata(self, tmp_path, checkpoints):
        out_dir = _run_recipe(tmp_path, checkpoints)

        assert _meta(out_dir) == {
            "vocab_size": 1024,
            "blank_id": 1024,
            "prompt_dictionary": {"es": 0, "en": 1, "pt": 2},
        }

    def test_blank_id_is_num_classes_with_blank_minus_one(self, tmp_path, checkpoints):
        out_dir = _run_recipe(tmp_path, checkpoints, _fake_model(blank_plus_one=42))

        assert _meta(out_dir)["blank_id"] == 41

    def test_missing_checkpoint_argument_is_required(self):
        with pytest.raises(SystemExit):
            main([])

    def test_default_out_dir_follows_the_recipe_artifacts_convention(
        self, tmp_path, monkeypatch, checkpoints
    ):
        # RECIPE_ARTIFACTS_DIR is a relative path resolved against cwd -- chdir
        # into an isolated tmp_path so exercising the real default writes
        # nothing outside this test's own sandbox.
        monkeypatch.chdir(tmp_path)
        checkpoint = tmp_path / "checkpoint.nemo"
        checkpoints.add(checkpoint, _fake_model())

        main(["--checkpoint", str(checkpoint)])

        assert (
            tmp_path / RECIPE_ARTIFACTS_DIR / "parakeet_prompt" / "prompt_kernel_cache.npz"
        ).exists()
