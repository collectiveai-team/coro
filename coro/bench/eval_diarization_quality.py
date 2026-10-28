r"""Pure-diarization DER comparison: NeMo Sortformer vs pyannote community-1.

Runs each Diarization Adapter's ``diarize_pcm`` directly over AMI Mix-Headset
audio (bypassing ASR so speaker quality is isolated), writes a diarization-only
hypothesis STM per item, and scores Diarization Error Rate by reusing the
Quality Benchmark scoring (``coro.bench.quality.score_item`` / ``DerStats``) and
the shared STM writers — not a parallel DER/STM implementation.

Unlike the main **Quality Benchmark**, whose Hypothesis STM comes from ASR
segment spans, this tool scores the diarizer's own timeline. That is the only
configuration in which a published diarization figure can be checked, because
ASR segmentation otherwise fixes missed-speech and false-alarm regardless of
which diarization model runs.

Usage — whole meetings, against the built-in AMI reference::

    coro-bench-diar --ami-root ../../amicorpus \
        --meetings IS1009a ES2004a TS3003a --out-dir /tmp/diar-eval

Usage — reproducing a model card's AMI figure. Cards report AMI at a 0 s collar
with overlap scored, against forced-alignment RTTMs, at a named latency tier;
all four have to be set together or the comparison is not like-for-like::

    coro-bench-diar --clips-dir .tmp/ami-clips \
        --ref-rttm-dir /path/to/diar-forced-alignment/AMI \
        --collar 0 --regions all --latency-tier very-high \
        --backends nemo --nemo-model nvidia/diar_streaming_sortformer_4spk-v2.1

The pyannote model is gated; provide a token via CORO_HF_TOKEN, HF_TOKEN, or
HUGGING_FACE_HUB_TOKEN with the community-1 user conditions accepted, and
install the optional extra (``uv sync --extra cpu --extra diar-pyannote``).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import time
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch  # noqa: TID253

from coro.backends.diarization import factory as diarization_factory
from coro.backends.diarization.nemo.postprocessing import preset_for_collar
from coro.backends.diarization.nemo.streaming import (
    applied_streaming_params,
    get_latency_tier_params,
)
from coro.bench.diar_eval_args import DiarEvalArgs, parse_diar_eval_args
from coro.bench.diar_workload import (
    DiarItem,
    items_from_clips_dir,
    items_from_meetings,
    materialize_rttm_references,
)
from coro.bench.models.quality import DerStats
from coro.bench.quality import score_item
from coro.bench.stm import speaker_timeline_to_stm
from coro.cli_support import out_console
from coro.core.logger import get_logger
from coro.settings import get_settings

log = get_logger(__name__)


@dataclass
class MeetingResult:
    """Per-meeting DER (one DerStats per region mode) and timing."""

    der_by_mode: dict[str, DerStats]
    audio_seconds: float
    diar_seconds: float
    rtf: float
    n_segments: int
    n_speakers_hyp: int


@dataclass
class BackendResult:
    """One backend's per-meeting results and combined-across-meetings DER."""

    per_meeting: dict[str, MeetingResult] = field(default_factory=dict)
    combined: dict[str, DerStats] = field(default_factory=dict)


def _read_pcm(wav_path: Path) -> tuple[bytes, float]:
    """Read a 16 kHz mono 16-bit WAV into raw PCM bytes and duration seconds."""
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == 16000, f"{wav_path}: expected 16 kHz"  # noqa: S101
        assert wav.getnchannels() == 1, f"{wav_path}: expected mono"  # noqa: S101
        assert wav.getsampwidth() == 2, f"{wav_path}: expected 16-bit"  # noqa: S101
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    return pcm, frames / 16000.0


def _score_der(ref_stm: Path, hyp_stm: Path, *, collar: float, regions: str) -> DerStats:
    """Score DER for one session by reusing the Quality Benchmark scoring."""
    result = score_item(ref_stm, hyp_stm, der_collar=collar, der_regions=regions)
    if result.metrics is None or result.metrics.der is None:
        error = result.error.message if result.error else "no DER produced"
        msg = f"DER scoring failed for {hyp_stm.name}: {error}"
        raise RuntimeError(msg)
    return result.metrics.der


def combine_der(per_item: list[DerStats]) -> DerStats:
    """Pool per-session DER into one duration-weighted figure.

    md-eval's three error terms and its denominator are all seconds, so pooling
    is a sum over sessions and a single division — the same thing
    ``combine_error_rates`` does for the Quality Benchmark. Computed here rather
    than by handing meeteval a directory: it does not accept one, so the
    previous directory-based combined pass raised ``IsADirectoryError`` and this
    tool never reported a combined figure at all.
    """
    totals = {
        "missed_detection": sum(d.missed_detection for d in per_item),
        "false_alarm": sum(d.false_alarm for d in per_item),
        "speaker_error": sum(d.speaker_error for d in per_item),
        "total_speech": sum(d.total_speech for d in per_item),
    }
    scored = totals["total_speech"]
    errors = totals["missed_detection"] + totals["false_alarm"] + totals["speaker_error"]
    return DerStats(der=errors / scored if scored else 0.0, **totals)


def _tier_scope(adapter, tier: str | None):
    """Apply a latency tier around each model call, or nothing when unset.

    Left unset, NeMo runs at whatever streaming configuration the checkpoint
    ships — for the streaming Sortformer revisions that is ``chunk_len=188``,
    ``chunk_right_context=1``, ``fifo_len=0``, an operating point none of the
    published latency rows correspond to. Reproducing a model card therefore
    requires naming the tier explicitly. Scoping is
    :func:`applied_streaming_params`' job: the model object is shared, so the
    tier must not outlive the call.
    """
    modules = getattr(getattr(adapter, "model", None), "sortformer_modules", None)
    if tier is None or modules is None:
        return contextlib.nullcontext()
    return applied_streaming_params(modules, get_latency_tier_params(tier))


def _run_backend(
    name: str,
    adapter,
    items: list[DiarItem],
    out_dir: Path,
    collar: float,
    region_modes: list[str],
    *,
    latency_tier: str | None = None,
) -> BackendResult:
    """Diarize every item with one adapter, score DER per region mode."""
    result = BackendResult()
    hyp_dir = out_dir / name / "hyp"
    ref_dir = out_dir / name / "ref"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        pcm, duration = _read_pcm(item.audio_path)

        t0 = time.perf_counter()
        with _tier_scope(adapter, latency_tier):
            timeline = asyncio.run(adapter.diarize_pcm(pcm))
        elapsed = time.perf_counter() - t0

        hyp_stm = hyp_dir / f"{item.item_id}.hyp.stm"
        hyp_stm.write_text(speaker_timeline_to_stm(timeline, item.item_id))
        # Copy the reference beside the hypothesis so the combined pass can read
        # one directory per arm, and so the run is self-describing afterwards.
        (ref_dir / f"{item.item_id}.ref.stm").write_text(item.ref_stm_path.read_text())

        der_by_mode = {
            mode: _score_der(item.ref_stm_path, hyp_stm, collar=collar, regions=mode)
            for mode in region_modes
        }
        n_spk = len({s.speaker for s in timeline})
        rtf = elapsed / duration if duration else 0.0
        result.per_meeting[item.item_id] = MeetingResult(
            der_by_mode=der_by_mode,
            audio_seconds=round(duration, 1),
            diar_seconds=round(elapsed, 1),
            rtf=round(rtf, 3),
            n_segments=len(timeline),
            n_speakers_hyp=n_spk,
        )
        primary = der_by_mode[region_modes[0]]
        modes_str = "  ".join(f"{m}={der_by_mode[m].der * 100:.1f}%" for m in region_modes)
        out_console.out(
            f"  [{name}] {item.item_id}: DER[{modes_str}]  "
            f"miss={primary.missed_detection:.0f}s fa={primary.false_alarm:.0f}s "
            f"spk_err={primary.speaker_error:.0f}s  hyp_spk={n_spk} segs={len(timeline)}  "
            f"({duration / 60:.1f}min in {elapsed:.0f}s, rtf={rtf:.2f})"
        )

    # Combined DER across all items, pooled from the per-item scores.
    for mode in region_modes:
        result.combined[mode] = combine_der(
            [m.der_by_mode[mode] for m in result.per_meeting.values()]
        )
    out_console.out(
        f"  [{name}] COMBINED DER "
        + "  ".join(f"{m}={result.combined[m].der * 100:.2f}%" for m in region_modes)
    )
    return result


def _free(adapter) -> None:
    """Drop an adapter and reclaim GPU memory before loading the next model."""
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_args(argv: Sequence[str] | None = None) -> DiarEvalArgs:
    return parse_diar_eval_args(argv, description=__doc__)


def resolve_workload(args: DiarEvalArgs, out_dir: Path) -> list[DiarItem]:
    """Build the item list and point it at the requested reference source."""
    if args.clips_dir:
        items = items_from_clips_dir(Path(args.clips_dir).resolve())
    else:
        items = items_from_meetings(args.meetings or [], Path(args.ami_root).resolve())
    if args.ref_rttm_dir:
        items = materialize_rttm_references(
            items, Path(args.ref_rttm_dir).resolve(), out_dir / "ref-rttm"
        )
    return items


def resolve_postprocessing_selection(value: str | None, *, collar: float) -> str | None:
    """Pick the Diarization Post-Processing Configuration for this scoring collar.

    Zero-collar scoring rewards boundary precision and near-zero padding, while
    collar-tolerant scoring rewards generous padding and aggressive
    short-segment deletion — so a parameter set tuned for one collar is the
    wrong set for the other. Defaulting to ``auto`` pairs this tool's
    ``--collar`` with the preset tuned for it instead of inheriting whichever
    default happens to apply. See ADR 0010.
    """
    if value == "none":
        return None
    if value is None or value == "auto":
        return preset_for_collar(collar)
    return value


def _print_tables(
    item_ids: list[str],
    ran: list[str],
    results: dict[str, BackendResult],
    *,
    collar: float,
    region_modes: list[str],
) -> None:
    width = max([10, *(len(i) for i in item_ids)])
    for mode in region_modes:
        out_console.out("\n" + "=" * 70)
        out_console.out(f"DIARIZATION QUALITY  (DER, collar={collar}s, regions={mode})")
        out_console.out("=" * 70)
        header = f"{'item':<{width}}" + "".join(f"{b:>14}" for b in ran)
        out_console.out(header)
        out_console.out("-" * len(header))
        for item_id in item_ids:
            row = f"{item_id:<{width}}"
            for b in ran:
                der = results[b].per_meeting[item_id].der_by_mode[mode].der
                row += f"{der * 100:>13.1f}%"
            out_console.out(row)
        out_console.out("-" * len(header))
        comb_row = f"{'COMBINED':<{width}}"
        for b in ran:
            comb_row += f"{results[b].combined[mode].der * 100:>13.2f}%"
        out_console.out(comb_row)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the diarization-only DER comparison (``coro-bench-diar``)."""
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token_secret = get_settings().hf_token
    hf_token = hf_token_secret.get_secret_value() if hf_token_secret else None

    region_modes = [str(mode) for mode in args.regions]
    postprocessing = resolve_postprocessing_selection(args.postprocessing, collar=args.collar)
    items = resolve_workload(args, out_dir)
    out_console.out(
        f"Items: {len(items)} ({', '.join(i.item_id for i in items)})\n"
        f"Reference: {args.ref_rttm_dir or 'built-in AMI Reference STM'}\n"
        f"Collar: {args.collar}  Regions: {region_modes}\n"
        f"Latency tier (nemo): {args.latency_tier or 'checkpoint default'}\n"
        f"Post-processing (nemo): {postprocessing or 'baseline (none)'}\n"
    )

    models = {"nemo": args.nemo_model, "pyannote": args.pyannote_model}
    results: dict[str, BackendResult] = {}
    errors: dict[str, str] = {}

    for name in args.backends:
        out_console.out(f"\n=== {name} ({models[name]}) ===")
        try:
            adapter = diarization_factory.build_diarization_adapter(
                name,
                models[name],
                device="auto",
                hf_token=hf_token,
                postprocessing=postprocessing,
            )
            results[name] = _run_backend(
                name,
                adapter,
                items,
                out_dir,
                args.collar,
                region_modes,
                latency_tier=args.latency_tier if name == "nemo" else None,
            )
            _free(adapter)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            log.exception("backend failed", backend=name)

    payload = {
        "protocol": {
            "collar": args.collar,
            "regions": region_modes,
            "reference": args.ref_rttm_dir or "built-in AMI Reference STM",
            "latency_tier": args.latency_tier or "checkpoint default",
            "postprocessing": postprocessing or "baseline (none)",
            "items": [i.item_id for i in items],
        },
        "results": {name: asdict(res) for name, res in results.items()},
        "errors": errors,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    ran = [b for b in args.backends if b in results]
    item_ids = [i.item_id for i in items]
    _print_tables(item_ids, ran, results, collar=args.collar, region_modes=region_modes)
    for b, msg in errors.items():
        out_console.out(f"\n[{b}] could not run: {msg}")
    out_console.out(f"\nFull results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
