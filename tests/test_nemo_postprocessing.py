"""Diarization Post-Processing Configuration resolution. See ADR 0010."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from coro.backends.diarization.nemo.postprocessing import (
    _PRESET_DIR,
    _PRESETS,
    apply_gated_postprocessing,
    baseline_postprocessing_params,
    estimate_speaker_count,
    postprocessing_gate_open,
    preset_for_collar,
    preset_provenance,
    resolve_postprocessing_yaml,
)


@pytest.mark.parametrize("value", [None, "", "none"])
def test_baseline_selectors_resolve_to_no_override(value):
    """None, empty and the explicit 'none' selector all keep NeMo's baseline.

    The explicit spellings matter because an operator templating the env var can
    only unset it by writing something: it must resolve, not fail Strict Startup
    Validation.
    """
    assert resolve_postprocessing_yaml(value) is None


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


# ---------------------------------------------------------------------------
# Vendored preset provenance and collar pairing
# ---------------------------------------------------------------------------


_UPSTREAM_SOURCE = (
    "NVIDIA-NeMo/Speech (Apache-2.0), examples/speaker_tasks/diarization/conf/post_processing/"
)


def test_every_preset_records_its_provenance_and_target_collar():
    """A parameter set is only usable if you know what it was tuned against.

    Pinned by value rather than by presence: provenance that drifts silently is
    the same as no provenance, and the collar is what selection keys on.
    """
    assert {
        name: (p.optimized_on, p.target_collar_s, p.source) for name, p in _PRESETS.items()
    } == {
        "dihard3-dev": ("DIHARD III dev split", 0.0, _UPSTREAM_SOURCE),
        "callhome-part1": ("CALLHOME (NIST SRE 2000 Disc8), part1", 0.25, _UPSTREAM_SOURCE),
    }


@pytest.mark.parametrize("preset_name", sorted(_PRESETS))
def test_vendored_yaml_keeps_upstream_attribution(preset_name):
    """Vendored verbatim, with NVIDIA's own attribution comments intact."""
    text = (_PRESET_DIR / _PRESETS[preset_name].filename).read_text()
    assert "NVIDIA-NeMo/Speech" in text
    assert "parameters:" in text


def test_preset_for_collar_pairs_zero_collar_with_the_zero_collar_set():
    """The main Quality Benchmark scores at collar 0.0."""
    assert preset_for_collar(0.0) == "dihard3-dev"


def test_preset_for_collar_pairs_quarter_second_with_the_tolerant_set():
    """coro-bench-diar defaults to collar 0.25."""
    assert preset_for_collar(0.25) == "callhome-part1"


def test_the_two_presets_target_different_collars():
    """If they agreed there would be nothing to select between."""
    collars = {p.target_collar_s for p in preset_provenance()}
    assert len(collars) > 1


# ---------------------------------------------------------------------------
# Speaker-count estimate
# ---------------------------------------------------------------------------


def _preds(n_active: int, *, n_spk: int = 8, frames: int = 200):
    preds = torch.zeros(frames, n_spk)
    for spk in range(n_active):
        preds[:, spk] = 0.99
    return preds


@pytest.mark.parametrize("n_active", [0, 1, 3, 4, 5, 8])
def test_estimate_counts_clearly_present_speakers(n_active):
    assert estimate_speaker_count(_preds(n_active)) == n_active


def test_estimate_accepts_batched_predictions():
    """Streaming holds (1, frames, speakers); batch hands over the same shape."""
    assert estimate_speaker_count(_preds(3).unsqueeze(0)) == 3


def test_estimate_ignores_a_brief_flicker():
    """A few isolated frames are not a speaker."""
    preds = _preds(2)
    preds[:2, 5] = 0.99  # 2 frames = 0.16s, below the 0.5s presence floor
    assert estimate_speaker_count(preds) == 2


def test_estimate_of_empty_predictions_is_zero():
    assert estimate_speaker_count(torch.zeros(0, 4)) == 0


# ---------------------------------------------------------------------------
# Speaker-Count Post-Processing Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_active", [1, 3, 4])
def test_gate_open_at_or_below_the_ceiling(n_active):
    gate_open, estimated = postprocessing_gate_open(_preds(n_active), max_speakers=4)
    assert gate_open is True
    assert estimated == n_active


@pytest.mark.parametrize("n_active", [5, 6, 8])
def test_gate_closed_above_the_ceiling(n_active):
    """NVIDIA reports tuned post-processing degrades DER at five or more speakers."""
    gate_open, estimated = postprocessing_gate_open(_preds(n_active), max_speakers=4)
    assert gate_open is False
    assert estimated == n_active


def test_gate_ceiling_is_configurable():
    """So the gate is already correct when a >4-speaker model is selected."""
    assert postprocessing_gate_open(_preds(6), max_speakers=8)[0] is True
    assert postprocessing_gate_open(_preds(6), max_speakers=4)[0] is False


def test_gate_cannot_close_on_a_four_speaker_model():
    """Documented limitation: a T x 4 matrix can never estimate above 4.

    This is why the gate is unobservable on every currently shipped Sortformer
    revision, and why it is built anyway. See ADR 0010.
    """
    all_four_active = _preds(4, n_spk=4)
    assert postprocessing_gate_open(all_four_active, max_speakers=4)[0] is True


def test_gated_postprocessing_drops_tuned_thresholds_when_closed():
    """Above the ceiling the tuned set must not be the one applied."""
    preds = _preds(6)
    tuned_path = resolve_postprocessing_yaml("callhome-part1")

    closed = apply_gated_postprocessing(
        preds, n_spk=8, postprocessing_yaml=tuned_path, max_speakers=4
    )
    baseline = apply_gated_postprocessing(preds, n_spk=8, postprocessing_yaml=None, max_speakers=4)
    tuned = apply_gated_postprocessing(
        preds, n_spk=8, postprocessing_yaml=tuned_path, max_speakers=8
    )

    # Gate closed => identical to the plain baseline, not to the tuned run.
    assert closed == baseline
    assert closed != tuned


def test_baseline_params_are_a_fresh_object_each_call():
    """NeMo mutates the params it is handed when bypassing; sharing one leaks."""
    first = baseline_postprocessing_params()
    first.onset = 0.123
    assert baseline_postprocessing_params().onset != 0.123
