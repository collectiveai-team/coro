"""Diarization Post-Processing Configuration resolution. See ADR 0009."""

from __future__ import annotations

from pathlib import Path

import pytest

from coro.backends.diarization.nemo.postprocessing import (
    _PRESET_DIR,
    _PRESETS,
    resolve_postprocessing_yaml,
)


def test_none_passes_through_unchanged():
    """None keeps NeMo's own unconfigured baseline — no override applied."""
    assert resolve_postprocessing_yaml(None) is None


@pytest.mark.parametrize("preset_name", sorted(_PRESETS))
def test_known_preset_resolves_to_vendored_file(preset_name):
    """Every registered preset name resolves to an existing vendored file."""
    resolved = resolve_postprocessing_yaml(preset_name)
    assert resolved is not None
    assert Path(resolved).is_file()
    assert Path(resolved).parent == _PRESET_DIR


def test_custom_path_resolves_when_file_exists(tmp_path):
    """A literal filesystem path is accepted when the file exists."""
    custom = tmp_path / "custom.yaml"
    custom.write_text("parameters:\n  onset: 0.5\n")

    resolved = resolve_postprocessing_yaml(str(custom))

    assert resolved == str(custom)


def test_unknown_value_that_is_not_a_file_raises():
    """A value that is neither a known preset nor an existing path fails loudly."""
    with pytest.raises(ValueError, match="known preset"):
        resolve_postprocessing_yaml("not-a-real-preset-or-path")


def test_nonexistent_custom_path_raises(tmp_path):
    """A path-shaped value that does not exist on disk fails loudly."""
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(ValueError, match="known preset"):
        resolve_postprocessing_yaml(str(missing))
