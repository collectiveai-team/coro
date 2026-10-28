"""The flag surface of ``coro-bench-diar``, one typed field per ``--kebab-case`` flag."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal

from cyclopts import Parameter

from coro.backends.diarization.nemo.streaming import LATENCY_TIER_PARAMS
from coro.cli_support import fail, invoke, new_app

PROG = "coro-bench-diar"

RegionMode = Literal["all", "nooverlap", "single"]

_Many = Parameter(consume_multiple=True)


def _check_latency_tier(type_: object, value: str | None) -> None:
    """Restrict ``--latency-tier`` to the tiers the NeMo streaming module defines."""
    if value is not None and value not in LATENCY_TIER_PARAMS:
        msg = f"choose from {', '.join(sorted(LATENCY_TIER_PARAMS))}"
        raise ValueError(msg)


@Parameter(name="*")
@dataclass(kw_only=True)
class DiarEvalArgs:
    """Parsed ``coro-bench-diar`` arguments."""

    ami_root: str = "../../amicorpus"
    meetings: Annotated[
        list[str] | None,
        Parameter(
            consume_multiple=True,
            help="Whole AMI meetings to score. Mutually exclusive with --clips-dir.",
        ),
    ] = None
    clips_dir: Annotated[
        str | None,
        Parameter(
            help="Directory of pre-cut clips (<clip>.wav beside <clip>.ref.stm, as "
            "make_ami_clip writes them). Mutually exclusive with --meetings."
        ),
    ] = None
    ref_rttm_dir: Annotated[
        str | None,
        Parameter(
            help="Score against external RTTMs instead of the built-in Reference STM. "
            "Flat or split into train/dev/test. Needed to check a published AMI "
            "figure: model cards score AMI against forced-alignment RTTMs, which "
            "disagree with the manual-annotation reference by roughly 29 DER points."
        ),
    ] = None
    latency_tier: Annotated[
        str | None,
        Parameter(
            validator=_check_latency_tier,
            help="Streaming latency tier to apply around each nemo model call. Unset "
            "runs the checkpoint's own configuration, which for the streaming "
            "Sortformer revisions is not one of the published latency rows; "
            "'very-high' is the 30.4 s row model cards report AMI at. Choices: "
            f"{', '.join(sorted(LATENCY_TIER_PARAMS))}.",
        ),
    ] = None
    collar: Annotated[
        float,
        Parameter(
            help="DER collar in seconds. Defaults to 0, the convention published AMI "
            "diarization results use; pass 0.25 for the telephony convention. "
            "This also selects the collar-matched post-processing preset."
        ),
    ] = 0.0
    postprocessing: Annotated[
        str | None,
        Parameter(
            help="Diarization Post-Processing Configuration for the nemo backend: a "
            "vendored preset name, a path to a custom YAML, 'none' for NeMo's "
            "unconfigured baseline, or 'auto' (the default) to select the "
            "vendored preset tuned for --collar. See ADR 0010."
        ),
    ] = None
    regions: Annotated[
        list[RegionMode],
        Parameter(
            consume_multiple=True,
            help="One or more DER region modes to score (first is the primary).",
        ),
    ] = field(default_factory=lambda: ["all", "nooverlap"])
    out_dir: str = "/tmp/diar-eval"  # noqa: S108
    nemo_model: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    backends: Annotated[list[str], _Many] = field(default_factory=lambda: ["nemo", "pyannote"])


def _command(args: DiarEvalArgs | None = None) -> DiarEvalArgs:
    return args if args is not None else DiarEvalArgs()


def parse_diar_eval_args(
    argv: Sequence[str] | None = None, *, description: str | None = None
) -> DiarEvalArgs:
    """Parse and validate ``coro-bench-diar`` arguments.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.
        description: Text shown at the top of ``--help``.

    Returns:
        The parsed arguments.

    Raises:
        SystemExit: Code 2 on invalid arguments, code 0 after ``--help``.

    """
    app = new_app(PROG, help=description)
    app.default(_command)
    args: DiarEvalArgs = invoke(app, argv)
    if bool(args.meetings) == bool(args.clips_dir):
        fail(PROG, "pass exactly one of --meetings or --clips-dir")
    return args
