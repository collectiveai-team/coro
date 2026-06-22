"""Pure-diarization DER comparison: NeMo Sortformer vs pyannote community-1.

Runs each Diarization Adapter's ``diarize_pcm`` directly over AMI Mix-Headset
audio (bypassing ASR so speaker quality is isolated), writes a diarization-only
hypothesis STM per meeting, and scores Diarization Error Rate by reusing the
Quality Benchmark scoring (``coro.bench.quality.score_item`` / ``DerStats``) and
the shared STM writers — not a parallel DER/STM implementation.

Usage:
    coro-bench-diar --ami-root ../../amicorpus \\
        --meetings IS1009a ES2004a TS3003a \\
        --collar 0.25 --regions all --out-dir /tmp/diar-eval

The pyannote model is gated; provide a token via CORO_HF_TOKEN, HF_TOKEN, or
HUGGING_FACE_HUB_TOKEN with the community-1 user conditions accepted, and
install the optional extra (``uv sync --extra cpu --extra diar-pyannote``).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from coro.backends.diarization import factory as diarization_factory
from coro.bench.models.quality import DerStats
from coro.bench.quality import score_item
from coro.bench.stm import speaker_timeline_to_stm
from coro.settings import ServerSettings


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


def _run_backend(
    name: str,
    adapter,
    meetings: list[str],
    ami_root: Path,
    out_dir: Path,
    collar: float,
    region_modes: list[str],
) -> BackendResult:
    """Diarize every meeting with one adapter, score DER per region mode."""
    result = BackendResult()
    hyp_dir = out_dir / name / "hyp"
    hyp_dir.mkdir(parents=True, exist_ok=True)

    for meeting_id in meetings:
        wav_path = ami_root / meeting_id / "audio" / f"{meeting_id}.Mix-Headset.wav"
        ref_stm = ami_root / "stm" / f"{meeting_id}.ref.stm"
        pcm, duration = _read_pcm(wav_path)

        t0 = time.perf_counter()
        timeline = asyncio.run(adapter.diarize_pcm(pcm))
        elapsed = time.perf_counter() - t0

        hyp_stm = hyp_dir / f"{meeting_id}.hyp.stm"
        hyp_stm.write_text(speaker_timeline_to_stm(timeline, meeting_id))

        der_by_mode = {
            mode: _score_der(ref_stm, hyp_stm, collar=collar, regions=mode)
            for mode in region_modes
        }
        n_spk = len({s.speaker for s in timeline})
        rtf = elapsed / duration if duration else 0.0
        result.per_meeting[meeting_id] = MeetingResult(
            der_by_mode=der_by_mode,
            audio_seconds=round(duration, 1),
            diar_seconds=round(elapsed, 1),
            rtf=round(rtf, 3),
            n_segments=len(timeline),
            n_speakers_hyp=n_spk,
        )
        primary = der_by_mode[region_modes[0]]
        modes_str = "  ".join(f"{m}={der_by_mode[m].der * 100:.1f}%" for m in region_modes)
        print(
            f"  [{name}] {meeting_id}: DER[{modes_str}]  "
            f"miss={primary.missed_detection:.0f}s fa={primary.false_alarm:.0f}s "
            f"spk_err={primary.speaker_error:.0f}s  hyp_spk={n_spk} segs={len(timeline)}  "
            f"({duration / 60:.1f}min in {elapsed:.0f}s, rtf={rtf:.2f})",
            flush=True,
        )

    # Combined DER across all meetings (re-score every session together) per mode.
    ref_all = ami_root / "stm"
    for mode in region_modes:
        # score_item handles multifile STM dirs; combine all hyp sessions in one pass.
        result.combined[mode] = _score_der(ref_all, hyp_dir, collar=collar, regions=mode)
    print(
        f"  [{name}] COMBINED DER "
        + "  ".join(f"{m}={result.combined[m].der * 100:.2f}%" for m in region_modes),
        flush=True,
    )
    return result


def _free(adapter) -> None:
    """Drop an adapter and reclaim GPU memory before loading the next model."""
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ami-root", default="../../amicorpus")
    parser.add_argument("--meetings", nargs="+", required=True)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["all", "nooverlap"],
        choices=["all", "nooverlap", "single"],
        help="One or more DER region modes to score (first is the primary).",
    )
    parser.add_argument("--out-dir", default="/tmp/diar-eval")  # noqa: S108
    parser.add_argument("--nemo-model", default="nvidia/diar_streaming_sortformer_4spk-v2")
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--backends", nargs="+", default=["nemo", "pyannote"])
    return parser.parse_args()


def _print_tables(
    meetings: list[str],
    ran: list[str],
    results: dict[str, BackendResult],
    *,
    collar: float,
    region_modes: list[str],
) -> None:
    for mode in region_modes:
        print("\n" + "=" * 70)
        print(f"DIARIZATION QUALITY  (DER, collar={collar}s, regions={mode})")
        print("=" * 70)
        header = f"{'meeting':<10}" + "".join(f"{b:>14}" for b in ran)
        print(header)
        print("-" * len(header))
        for meeting_id in meetings:
            row = f"{meeting_id:<10}"
            for b in ran:
                der = results[b].per_meeting[meeting_id].der_by_mode[mode].der
                row += f"{der * 100:>13.1f}%"
            print(row)
        print("-" * len(header))
        comb_row = f"{'COMBINED':<10}"
        for b in ran:
            comb_row += f"{results[b].combined[mode].der * 100:>13.2f}%"
        print(comb_row)


def main() -> None:
    args = _parse_args()
    ami_root = Path(args.ami_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token_secret = ServerSettings().hf_token
    hf_token = hf_token_secret.get_secret_value() if hf_token_secret else None

    region_modes = list(args.regions)
    print(
        f"AMI root: {ami_root}\nMeetings: {args.meetings}\n"
        f"Collar: {args.collar}  Regions: {region_modes}\n"
    )

    models = {"nemo": args.nemo_model, "pyannote": args.pyannote_model}
    results: dict[str, BackendResult] = {}
    errors: dict[str, str] = {}

    for name in args.backends:
        print(f"\n=== {name} ({models[name]}) ===", flush=True)
        try:
            adapter = diarization_factory.build_diarization_adapter(
                name, models[name], device="auto", hf_token=hf_token
            )
            results[name] = _run_backend(
                name, adapter, args.meetings, ami_root, out_dir, args.collar, region_modes
            )
            _free(adapter)
        except Exception as exc:  # noqa: BLE001 - surface per-backend failure, keep going
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"  [{name}] FAILED: {errors[name]}", flush=True)

    payload = {
        "results": {name: asdict(res) for name, res in results.items()},
        "errors": errors,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    ran = [b for b in args.backends if b in results]
    _print_tables(args.meetings, ran, results, collar=args.collar, region_modes=region_modes)
    for b, msg in errors.items():
        print(f"\n[{b}] could not run: {msg}")
    print(f"\nFull results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
