"""Tests for WDER — Word Diarization Error Rate.

Covers the degenerate cases the metric must get right, the properties that
motivate its existence (segmentation blindness, speaker-label invariance), and
the requirement that the speaker mapping comes from meeteval's cpWER assignment
rather than being recomputed here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from coro.bench.models.quality import WderStats
from coro.bench.wder import (
    UNKNOWN_SPEAKER_LABEL,
    combine_wder,
    compute_wder,
    hyp_to_ref_speaker_map,
)
from support.wder_stm import REF, Line

meeteval = pytest.importorskip("meeteval")


def _score(ref: Path, hyp: Path) -> WderStats:
    return compute_wder(ref, hyp, meeteval.wer.cpwer(ref, hyp))


class TestDegenerateCases:
    def test_perfect_attribution_scores_zero(self, stm_pair):
        ref, hyp = stm_pair(
            REF,
            [
                ("S1", 0.0, 1.0, "alpha"),
                ("S2", 1.0, 2.0, "bravo"),
                ("S1", 2.0, 3.0, "charlie"),
                ("S2", 3.0, 4.0, "delta"),
            ],
        )
        stats = _score(ref, hyp)
        assert stats.wder == 0.0
        assert stats.wder_claimed == 0.0
        assert stats.abstention_rate == 0.0
        assert stats.scored == 4

    def test_maximally_wrong_attribution_scores_one(self, stm_pair):
        """Every scored word on a stream no reference speaker maps to."""
        ref, hyp = stm_pair(REF, [(UNKNOWN_SPEAKER_LABEL, 0.0, 4.0, "alpha bravo charlie delta")])
        stats = _score(ref, hyp)
        assert stats.wder == 1.0
        assert stats.speaker_errors == stats.scored == 4

    def test_all_unknown_hypothesis_abstains_rather_than_scoring_zero(self, stm_pair):
        """wder_claimed is undefined, not 0.0 — abstaining is not precision."""
        ref, hyp = stm_pair(REF, [(UNKNOWN_SPEAKER_LABEL, 0.0, 4.0, "alpha bravo charlie delta")])
        stats = _score(ref, hyp)
        assert stats.abstention_rate == 1.0
        assert stats.wder_claimed is None
        assert stats.claimed == 0

    def test_unknown_sentinel_is_an_error_even_when_meeteval_matches_it(self, stm_pair):
        """cpWER may pair `-1` with a reference speaker; WDER must not accept it.

        The sentinel means "no diarization support", so crediting it because the
        Hungarian permutation happened to have a free slot would report coverage
        the system never claimed.
        """
        ref, hyp = stm_pair(REF, [(UNKNOWN_SPEAKER_LABEL, 0.0, 4.0, "alpha bravo charlie delta")])
        cp_result = next(iter(meeteval.wer.cpwer(ref, hyp).values()))
        assert UNKNOWN_SPEAKER_LABEL in [h for _, h in cp_result.assignment]
        assert UNKNOWN_SPEAKER_LABEL not in hyp_to_ref_speaker_map(cp_result).pairs

    def test_empty_scored_population_yields_undefined_rates(self, stm_pair):
        """No word survives alignment as correct or substituted."""
        ref, hyp = stm_pair([("A", 0.0, 1.0, "alpha")], [("S1", 0.0, 1.0, "alpha")])
        stats = compute_wder(ref, hyp, {})
        assert stats.scored == 0
        assert stats.wder is None
        assert stats.wder_claimed is None
        assert stats.abstention_rate is None


class TestDiscrimination:
    def test_one_misattributed_word_moves_the_metric(self, stm_pair):
        ref, hyp = stm_pair(
            REF,
            [
                ("S1", 0.0, 1.0, "alpha"),
                ("S2", 1.0, 2.0, "bravo"),
                ("S2", 2.0, 3.0, "charlie"),
                ("S2", 3.0, 4.0, "delta"),
            ],
        )
        assert _score(ref, hyp).wder == pytest.approx(0.25, abs=1e-12)

    def test_collapsing_every_word_onto_one_speaker_is_penalised(self, stm_pair):
        ref, hyp = stm_pair(REF, [("S1", 0.0, 4.0, "alpha bravo charlie delta")])
        assert _score(ref, hyp).wder == pytest.approx(0.5, abs=1e-12)

    def test_speaker_relabelling_is_free_by_design(self, stm_pair):
        """A consistent permutation of speaker names scores 0, not 1.

        Speaker labels are arbitrary names, and the cpWER assignment exists
        precisely to quotient them out. A "fully permuted" hypothesis is
        therefore *correct* attribution under different names; only attribution
        that no global permutation can repair is an error.
        """
        ref, hyp = stm_pair(
            REF,
            [
                ("zzz", 0.0, 1.0, "alpha"),
                ("aaa", 1.0, 2.0, "bravo"),
                ("zzz", 2.0, 3.0, "charlie"),
                ("aaa", 3.0, 4.0, "delta"),
            ],
        )
        assert _score(ref, hyp).wder == 0.0

    def test_resegmentation_alone_moves_the_metric_by_exactly_zero(self, stm_pair):
        """The property cpWER lacks: same word labels, different chunking."""
        merged: list[Line] = [
            ("S1", 0.0, 1.0, "alpha"),
            ("S2", 1.0, 2.0, "bravo"),
            ("S1", 2.0, 4.0, "charlie"),
            ("S2", 3.0, 4.0, "delta"),
        ]
        split: list[Line] = [
            ("S1", 0.0, 0.5, "alpha"),
            ("S2", 1.0, 1.5, "bravo"),
            ("S1", 2.0, 2.5, "charlie"),
            ("S2", 3.0, 3.5, "delta"),
        ]
        ref_a, hyp_a = stm_pair(REF, merged)
        merged_stats = _score(ref_a, hyp_a)
        ref_b, hyp_b = stm_pair(REF, split)
        split_stats = _score(ref_b, hyp_b)
        assert merged_stats.wder == split_stats.wder


class TestSpeakerMapping:
    def test_mapping_is_taken_from_meeteval_not_recomputed(self, stm_pair):
        """compute_wder must consume the supplied assignment verbatim.

        Feeding a deliberately wrong assignment has to change the result; if it
        does not, the mapping is being derived locally and the acceptance
        criterion is violated.
        """
        ref, hyp = stm_pair(
            REF,
            [
                ("S1", 0.0, 1.0, "alpha"),
                ("S2", 1.0, 2.0, "bravo"),
                ("S1", 2.0, 3.0, "charlie"),
                ("S2", 3.0, 4.0, "delta"),
            ],
        )
        genuine = meeteval.wer.cpwer(ref, hyp)
        assert compute_wder(ref, hyp, genuine).wder == 0.0

        session_id, cp_result = next(iter(genuine.items()))
        swapped = {session_id: _with_assignment(cp_result, (("A", "S2"), ("B", "S1")))}
        assert compute_wder(ref, hyp, swapped).wder == 1.0

    def test_no_permutation_solver_is_invoked(self, stm_pair):
        """Guards against silently reintroducing a local Hungarian solve."""
        ref, hyp = stm_pair(REF, [("S1", 0.0, 4.0, "alpha bravo charlie delta")])
        cp_results = meeteval.wer.cpwer(ref, hyp)
        with patch("scipy.optimize.linear_sum_assignment", autospec=True) as solver:
            compute_wder(ref, hyp, cp_results)
        solver.assert_not_called()

    def test_unmatched_streams_are_dropped_from_the_mapping(self):
        cp_result = _with_assignment(None, (("A", "S1"), ("B", None), (None, "S2")))
        speakers = hyp_to_ref_speaker_map(cp_result)
        assert speakers.pairs == {"S1": "A"}
        assert speakers.matches("S1", "A") is True
        assert speakers.matches("S2", "B") is False


class TestCombine:
    def test_counts_pool_rather_than_rates_averaging(self):
        small = WderStats(
            wder=1.0,
            wder_claimed=1.0,
            abstention_rate=0.0,
            scored=1,
            speaker_errors=1,
            claimed=1,
            claimed_speaker_errors=1,
            abstentions=0,
            correct=1,
            substitutions=0,
        )
        large = WderStats(
            wder=0.0,
            wder_claimed=0.0,
            abstention_rate=0.0,
            scored=99,
            speaker_errors=0,
            claimed=99,
            claimed_speaker_errors=0,
            abstentions=0,
            correct=99,
            substitutions=0,
        )
        combined = combine_wder([small, large])
        assert combined is not None
        # Averaging the rates would give 0.5; pooling the counts gives 0.01.
        assert combined.wder == pytest.approx(0.01, abs=1e-12)
        assert combined.scored == 100

    def test_no_results_yields_none(self):
        assert combine_wder([]) is None


class TestDecomposition:
    def test_three_numbers_satisfy_the_precision_coverage_identity(self, stm_pair):
        """wder = wder_claimed * (1 - abstention_rate) + abstention_rate."""
        ref, hyp = stm_pair(
            REF,
            [
                ("S1", 0.0, 1.0, "alpha"),
                (UNKNOWN_SPEAKER_LABEL, 1.0, 2.0, "bravo"),
                ("S2", 2.0, 3.0, "charlie"),
                ("S2", 3.0, 4.0, "delta"),
            ],
        )
        stats = _score(ref, hyp)
        assert stats.wder_claimed is not None
        assert stats.abstention_rate is not None
        reconstructed = stats.wder_claimed * (1 - stats.abstention_rate) + stats.abstention_rate
        assert stats.wder == pytest.approx(reconstructed, abs=1e-12)


def _with_assignment(template, assignment):
    """Build a stand-in cpWER result carrying only the assignment field."""

    class _Stub:
        pass

    stub = _Stub()
    stub.assignment = assignment
    return stub
