"""The flag surface of the coro-bench CLI, one typed field per ``--kebab-case`` flag.

Cyclopts derives the parser from :class:`BenchArgs`; validation and the
environment-backed defaults live in :mod:`coro.bench.cli_args`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Group, Parameter

from coro.bench.calibration import DEFAULT_CALIBRATION_MARGIN
from coro.bench.process_lookup import DEFAULT_SERVER_MATCH
from coro.bench.settings import get_settings
from coro.bench.spanish import SPANISH_PRESETS

MANAGED_GROUP = Group("bench-managed server", sort_key=1)
ATTACHED_GROUP = Group("bench-attached server", sort_key=2)
SPANISH_GROUP = Group("Spanish workload set (public corpora)", sort_key=3)

AmiGroup = Literal["IB", "IN", "ES", "IS", "TS", "EN"]
DerRegions = Literal["all", "nooverlap", "single"]


def _check_spanish_preset(type_: object, value: str | None) -> None:
    """Restrict ``--spanish-preset`` to the presets :mod:`coro.bench.spanish` defines."""
    if value is not None and value not in SPANISH_PRESETS:
        choices = ", ".join(sorted(SPANISH_PRESETS))
        msg = f"choose from {choices}"
        raise ValueError(msg)


def _default_out_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return get_settings().out_dir or Path(f"/tmp/asr-bench-{stamp}")  # noqa: S108


_Many = Parameter(consume_multiple=True)


@Parameter(name="*")
@dataclass(kw_only=True)
class BenchArgs:
    """Parsed coro-bench arguments."""

    subcommand: Annotated[str, Parameter(parse=False)] = ""
    cli_args: Annotated[list[str], Parameter(parse=False)] = field(default_factory=list)

    out_dir: Path = field(default_factory=_default_out_dir)
    reps: int = 1
    sample_interval: float = field(default_factory=lambda: get_settings().sample_interval)
    ami_meetings: Annotated[list[str], _Many] = field(default_factory=list)
    ami_groups: Annotated[list[AmiGroup], _Many] = field(default_factory=list)
    ami_preset: Literal["sample", "eval", "full"] | None = None
    ami_root: Path = Path("./amicorpus/")
    no_download: bool = False
    reuse_reference_stms: Annotated[
        bool,
        Parameter(
            help="Reuse Reference STM files already present under <ami-root>/stm instead "
            "of regenerating them. Opt-in: reused files are frozen against whatever the "
            "STM builder looked like when they were first written."
        ),
    ] = False

    server_asr_backend: Annotated[str, Parameter(group=MANAGED_GROUP)] = "faster-whisper"
    server_asr_model: Annotated[str, Parameter(group=MANAGED_GROUP)] = "openai/whisper-medium"
    server_diar_backend: Annotated[str, Parameter(group=MANAGED_GROUP)] = "nemo"
    server_diar_model: Annotated[str, Parameter(group=MANAGED_GROUP)] = (
        "nvidia/diar_streaming_sortformer_4spk-v2"
    )
    server_pipeline: Annotated[str, Parameter(group=MANAGED_GROUP)] = "full-memory"
    no_diarization: Annotated[bool, Parameter(group=MANAGED_GROUP)] = False
    server_port: Annotated[int, Parameter(group=MANAGED_GROUP)] = 0

    server_url: Annotated[str | None, Parameter(group=ATTACHED_GROUP)] = None
    server_pid: Annotated[
        int | None,
        Parameter(
            group=ATTACHED_GROUP,
            help="Root PID of the Server Process Tree to sample. Bench-attached only; "
            "a Bench-Managed Server reports its own PID.",
        ),
    ] = field(default_factory=lambda: get_settings().server_pid)
    server_match: Annotated[
        str | None,
        Parameter(
            group=ATTACHED_GROUP,
            help="Command-line substring identifying the Server Process Tree when "
            f"--server-pid is not given (default: {DEFAULT_SERVER_MATCH!r}). "
            "Bench-attached only.",
        ),
    ] = field(default_factory=lambda: get_settings().server_match)

    warmup: bool = False
    warmup_audio: Path | None = None
    audio: Path | None = None
    reference_stm: Path | None = None
    clips_dir: Annotated[
        Path | None,
        Parameter(
            help="Directory of (<stem>.wav, <stem>.ref.stm) pairs to benchmark as a "
            "short-clip / curated workload (e.g. make_ami_clip output)."
        ),
    ] = None
    der_collar: float = 0.0
    der_regions: DerRegions = "all"
    language: Annotated[
        str | None,
        Parameter(
            help="Language hint sent with every transcription request (for example es, "
            "es-US, or es-ES). Omit it to benchmark automatic language detection."
        ),
    ] = None
    stream: bool = False
    deepgram: Annotated[
        bool,
        Parameter(
            help="Send audio to the Deepgram-native /v1/listen endpoint instead of the "
            "OpenAI one. This is the only wire surface that carries per-word speaker "
            "labels, so WDER scores per-word attribution rather than the segment-level "
            "majority summary. Allowed on 'quality', unlike --stream."
        ),
    ] = False

    spanish_preset: Annotated[
        str | None,
        Parameter(
            group=SPANISH_GROUP,
            validator=_check_spanish_preset,
            help="Materialise a Spanish Workload Set from freely-licensed public "
            "corpora and benchmark it as a --clips-dir workload. Single-speaker: "
            f"WER only, no meaningful DER. Choices: {', '.join(sorted(SPANISH_PRESETS))}.",
        ),
    ] = None
    spanish_root: Annotated[Path, Parameter(group=SPANISH_GROUP)] = Path("./spanish-corpora/")
    spanish_limit: Annotated[
        int | None,
        Parameter(group=SPANISH_GROUP, help="Override the preset's items-per-corpus count."),
    ] = None
    spanish_fetch_plan: Annotated[
        bool,
        Parameter(
            group=SPANISH_GROUP,
            help="Print the download footprint of --spanish-preset and exit.",
        ),
    ] = False
    calibration_margin: Annotated[
        float,
        Parameter(
            group=SPANISH_GROUP,
            help="Two-sided absolute WER band against published figures "
            f"(default {DEFAULT_CALIBRATION_MARGIN}). Exceeding it fails the run.",
        ),
    ] = DEFAULT_CALIBRATION_MARGIN
    no_calibration: Annotated[
        bool,
        Parameter(
            group=SPANISH_GROUP,
            help="Report calibration without failing the run on a deviation.",
        ),
    ] = False
