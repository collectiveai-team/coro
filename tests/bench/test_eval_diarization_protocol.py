"""The scoring protocol knobs `coro-bench-diar` must be able to express.

These pin the configuration in which a published diarization figure can be
checked. Each one was found missing while resolving roadmap issue 16, where a
vendor AMI number could not be reproduced because the tool could not name the
collar, the reference or the latency tier the vendor used.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import fields

import pytest

from coro.backends.diarization.nemo.streaming import (
    LATENCY_TIER_PARAMS,
    LatencyTierParams,
    get_latency_tier_params,
)
from coro.bench.eval_diarization_quality import _parse_args, _tier_scope


def _args(*argv: str):
    original = sys.argv
    sys.argv = ["coro-bench-diar", *argv]
    try:
        return _parse_args()
    finally:
        sys.argv = original


def test_collar_defaults_to_zero():
    """Published AMI diarization results score at a 0 s collar."""
    assert _args("--meetings", "IS1009a").collar == 0.0


def test_a_workload_source_is_required():
    with pytest.raises(SystemExit):
        _args()


def test_help_renders(capsys):
    """argparse interpolates % in help strings; an unescaped one crashes --help."""
    with pytest.raises(SystemExit) as exc:
        _args("--help")

    assert exc.value.code == 0
    assert "--ref-rttm-dir" in capsys.readouterr().out


def test_meetings_and_clips_dir_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _args("--meetings", "IS1009a", "--clips-dir", "/tmp/clips")


def test_very_high_tier_is_the_configuration_model_cards_report_ami_at():
    """CHUNK 340 / RC 40 / FIFO 40 / UPDATE 300 / CACHE 188, i.e. 30.4 s latency."""
    params = get_latency_tier_params("very-high")

    assert (params.chunk_len, params.chunk_right_context) == (340, 40)
    assert (params.fifo_len, params.spkcache_update_period) == (40, 300)
    assert params.spkcache_len == 188
    latency_seconds = (params.chunk_len + params.chunk_right_context) * 0.08
    assert latency_seconds == pytest.approx(30.4, abs=0.01)


@pytest.mark.parametrize("tier", sorted(LATENCY_TIER_PARAMS))
def test_every_known_tier_is_accepted(tier: str):
    assert _args("--meetings", "m", "--latency-tier", tier).latency_tier == tier


def test_latency_tier_is_unset_by_default():
    """Unset means the checkpoint's configuration, which is not a published row."""
    assert _args("--meetings", "m").latency_tier is None


def test_latency_tier_rejects_an_unknown_tier():
    with pytest.raises(SystemExit):
        _args("--meetings", "m", "--latency-tier", "medium")


CHECKPOINT_DEFAULT = LatencyTierParams(
    chunk_len=188,
    chunk_right_context=1,
    fifo_len=0,
    spkcache_update_period=188,
    spkcache_len=188,
)
"""What the streaming Sortformer checkpoints ship — not a published latency row."""


class _Modules:
    """Stand-in for the shared ``sortformer_modules`` the tier is written onto."""

    def __init__(self) -> None:
        for field in fields(CHECKPOINT_DEFAULT):
            setattr(self, field.name, getattr(CHECKPOINT_DEFAULT, field.name))

    def snapshot(self) -> LatencyTierParams:
        return LatencyTierParams(
            **{field.name: getattr(self, field.name) for field in fields(CHECKPOINT_DEFAULT)}
        )


class _Adapter:
    def __init__(self, modules) -> None:
        self.model = type("Model", (), {"sortformer_modules": modules})()


def test_tier_scope_applies_the_tier_and_restores_the_shared_model():
    """The model object is shared with the batch adapter; the tier must not outlive the call."""
    modules = _Modules()

    with _tier_scope(_Adapter(modules), "very-high"):
        assert modules.snapshot() == get_latency_tier_params("very-high")

    assert modules.snapshot() == CHECKPOINT_DEFAULT


def test_tier_scope_leaves_the_checkpoint_configuration_alone_when_no_tier_is_requested():
    """Unset means the checkpoint's own configuration, not a silently applied default."""
    modules = _Modules()

    with _tier_scope(_Adapter(modules), None):
        assert modules.snapshot() == CHECKPOINT_DEFAULT


def test_tier_scope_tolerates_a_backend_without_sortformer_modules():
    """pyannote has no equivalent; asking for a tier must yield an inert scope."""
    scope = _tier_scope(object(), "very-high")

    assert isinstance(scope, contextlib.nullcontext)
    with scope:
        pass
