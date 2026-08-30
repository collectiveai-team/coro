"""coro-bench-diar pairs its scoring collar with a matched parameter set.

The main Quality Benchmark scores DER at collar 0.0 while this standalone
comparison tool defaults to 0.25. A parameter set tuned for one collar is the
wrong set for the other, so the pairing is made explicitly rather than
inherited. See ADR 0010.
"""

from __future__ import annotations

import pytest

from coro.bench.eval_diarization_quality import resolve_postprocessing_selection


def test_default_pairs_the_tools_own_quarter_second_collar():
    """coro-bench-diar's own --collar default is 0.25."""
    assert resolve_postprocessing_selection(None, collar=0.25) == "callhome-part1"


def test_auto_at_zero_collar_matches_the_main_harness_lane():
    """The main Quality Benchmark scores at collar 0.0."""
    assert resolve_postprocessing_selection("auto", collar=0.0) == "dihard3-dev"


def test_the_two_collars_do_not_select_the_same_set():
    """If they did, the collar-matching would be doing nothing."""
    assert resolve_postprocessing_selection("auto", collar=0.0) != (
        resolve_postprocessing_selection("auto", collar=0.25)
    )


def test_none_opts_out_to_the_nemo_baseline():
    assert resolve_postprocessing_selection("none", collar=0.25) is None


@pytest.mark.parametrize("explicit", ["dihard3-dev", "callhome-part1", "/tmp/custom.yaml"])
def test_explicit_selection_is_never_overridden_by_the_collar(explicit):
    """An operator who names a set gets that set, mismatched collar or not."""
    assert resolve_postprocessing_selection(explicit, collar=0.0) == explicit
