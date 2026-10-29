"""MeetEval adapter for Quality Benchmark scoring.

``meeteval`` lives in the optional ``bench`` dependency group, so it is imported
lazily through :func:`require_meeteval`, which exits with an install hint when
it is missing. The Protocols below type the part of its API the benchmark calls.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, SupportsFloat, cast, overload

from coro.bench.models.quality import DerStats, WerStats
from coro.cli_support import err_console


class ErrorRate(Protocol):
    """A meeteval WER result (``meeteval.wer.ErrorRate``)."""

    @property
    def error_rate(self) -> float: ...
    @property
    def errors(self) -> int: ...
    @property
    def length(self) -> int: ...
    @property
    def insertions(self) -> int: ...
    @property
    def deletions(self) -> int: ...
    @property
    def substitutions(self) -> int: ...


class CPErrorRate(ErrorRate, Protocol):
    """A meeteval cpWER result, which also carries the optimal speaker mapping."""

    @property
    def assignment(self) -> Sequence[tuple[str | None, str | None]] | None: ...


class DiaErrorRate(Protocol):
    """A meeteval DER result (``meeteval.der.md_eval.DiaErrorRate``)."""

    @property
    def error_rate(self) -> SupportsFloat: ...
    @property
    def scored_speaker_time(self) -> SupportsFloat: ...
    @property
    def missed_speaker_time(self) -> SupportsFloat: ...
    @property
    def falarm_speaker_time(self) -> SupportsFloat: ...
    @property
    def speaker_error_time(self) -> SupportsFloat: ...


type CPErrorRateTable = dict[str, CPErrorRate]
"""Per-session cpWER results, keyed by session id."""

type ErrorRateTable = dict[str, ErrorRate]
"""Per-session WER results, keyed by session id."""

type DiaErrorRateTable = dict[str, DiaErrorRate]
"""Per-session DER results, keyed by session id."""


class MeetevalWer(Protocol):
    """The ``meeteval.wer`` functions the benchmark calls."""

    def cpwer(self, reference: Path, hypothesis: Path) -> CPErrorRateTable: ...

    def greedy_orcwer(self, reference: Path, hypothesis: Path) -> ErrorRateTable: ...

    def greedy_dicpwer(self, reference: Path, hypothesis: Path) -> ErrorRateTable: ...

    # meeteval annotates only ErrorRate, but the function sums whatever rates it
    # is given, and DiaErrorRate supports that sum.
    @overload
    def combine_error_rates(self, *error_rates: DiaErrorRate) -> DiaErrorRate: ...
    @overload
    def combine_error_rates(self, *error_rates: ErrorRate) -> ErrorRate: ...


class MeetevalDer(Protocol):
    """The ``meeteval.der`` functions the benchmark calls."""

    def md_eval_22(
        self,
        reference: Path,
        hypothesis: Path,
        collar: float = 0,
        regions: str = "all",
    ) -> DiaErrorRateTable: ...


class Meeteval(Protocol):
    """The imported ``meeteval`` package, as far as the benchmark uses it."""

    @property
    def wer(self) -> MeetevalWer: ...

    @property
    def der(self) -> MeetevalDer: ...


def require_meeteval() -> Meeteval:
    """Import ``meeteval`` or exit with an install hint when it is missing."""
    try:
        import meeteval
    except ImportError:
        err_console.out(
            "Error: meeteval is required for quality scoring.\n"
            "The bench tooling is a PEP 735 dependency group, not an installable "
            "extra.\n"
            "Install with: uv sync --group bench\n"
            "Then run the bench with: uv run --group bench coro-bench ..."
        )
        sys.exit(1)
    # meeteval's own annotations are looser than its behaviour (assignment
    # elements, DER summing), so the module is read through the Protocols
    # above, which mirror meeteval 0.4.x.
    return cast("Meeteval", meeteval)  # botica: ignore[no-type-cast] -- see comment above


def wer_stats(result: ErrorRate) -> WerStats:
    """Convert a meeteval WER result object to :class:`WerStats`.

    meeteval 0.4.x uses `error_rate` instead of `wer` as the attribute name.
    """
    return WerStats(
        wer=result.error_rate,
        errors=result.errors,
        length=result.length,
        insertions=result.insertions,
        deletions=result.deletions,
        substitutions=result.substitutions,
    )


def der_stats(result: DiaErrorRate) -> DerStats:
    """Convert a meeteval DER result object to :class:`DerStats`.

    meeteval 0.4.x DiaErrorRate uses `*_speaker_time` field names and
    returns Decimal values; cast to float for JSON serialisability.
    """
    return DerStats(
        der=float(result.error_rate),
        false_alarm=float(result.falarm_speaker_time),
        missed_detection=float(result.missed_speaker_time),
        speaker_error=float(result.speaker_error_time),
        total_speech=float(result.scored_speaker_time),
    )


def combine_wer_sessions(meeteval: Meeteval, results: Mapping[str, ErrorRate]) -> ErrorRate:
    """Combine per-session meeteval WER results into a single aggregate."""
    return meeteval.wer.combine_error_rates(*results.values())


def combine_der_sessions(meeteval: Meeteval, results: Mapping[str, DiaErrorRate]) -> DiaErrorRate:
    """Combine per-session meeteval DER results into a single aggregate."""
    return meeteval.wer.combine_error_rates(*results.values())
