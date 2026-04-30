#!/usr/bin/env python3
"""ASR endpoint benchmark tool."""

import argparse
import csv
import json
import os
import re
import string
import subprocess
import sys
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

RESOURCE_FIELDNAMES = [
    "ts_epoch",
    "elapsed_s",
    "sample_dt_s",
    "root_pid",
    "process_count",
    "new_pids",
    "gone_pids",
    "rss_kb",
    "pss_kb",
    "uss_kb",
    "vsz_kb",
    "cpu_user_s",
    "cpu_system_s",
    "cpu_total_s",
    "cpu_pct",
    "thread_count",
    "io_rchar_bytes",
    "io_wchar_bytes",
    "io_read_bytes",
    "io_write_bytes",
    "io_rchar_bps",
    "io_wchar_bps",
    "io_read_bps",
    "io_write_bps",
    "server_vram_mib",
    "total_gpu_mem_mib",
    "total_gpu_used_mib",
    "gpu_util_pct",
    "observed_hardware_profile",
    "audio_seconds",
    "wall_seconds",
    "transcription_throughput",
    "wer",
    "der",
    "der_collar_s",
    "der_skip_overlap",
    "wer_normalization",
    "sampling_warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an ASR HTTP endpoint across N repetitions."
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("mode", nargs="?", default="json", choices=["json", "sse"],
                        help="Response mode (default: json)")
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of repetitions (default: 1)")
    parser.add_argument(
        "--url",
        default=os.environ.get("URL", "http://localhost:8000/v1/audio/transcriptions"),
        help="Endpoint URL",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=int(os.environ["SERVER_PID"]) if os.environ.get("SERVER_PID") else None,
        help="Server process PID",
    )
    parser.add_argument(
        "--server-match",
        default=os.environ.get("SERVER_MATCH", "custom_server.py"),
        help="pgrep pattern to find server PID",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "OUT_DIR",
            f"/tmp/asr-bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("SAMPLE_INTERVAL", "0.25")),
        help="Resource sampling interval in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--reference-transcript",
        type=Path,
        help="Reference transcript text file for WER scoring",
    )
    parser.add_argument(
        "--reference-diarization",
        type=Path,
        help="Reference RTTM file for DER scoring",
    )
    parser.add_argument(
        "--der-collar",
        type=float,
        default=0.25,
        help="DER collar in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--der-skip-overlap",
        action="store_true",
        help="Skip overlapped speech when scoring DER (default: include overlap)",
    )
    return parser.parse_args()


def find_server_pid(server_pid: int | None, server_match: str) -> int:
    if server_pid is not None:
        try:
            os.kill(server_pid, 0)  # signal 0 = existence check
        except PermissionError:
            pass  # process exists but owned by another user
        except ProcessLookupError:
            print(f"Error: server PID {server_pid} is not running", file=sys.stderr)
            sys.exit(1)
        return server_pid

    result = subprocess.run(
        ["pgrep", "-f", server_match],
        capture_output=True,
        text=True,
    )
    pids = result.stdout.strip().splitlines()
    if not pids:
        print(
            f"Error: could not find server process matching '{server_match}'. "
            "Set --server-pid or --server-match.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(pids) > 1:
        print(
            f"Warning: found {len(pids)} processes matching '{server_match}', "
            f"using PID {pids[0]}. Use --server-pid to disambiguate.",
            file=sys.stderr,
        )
    pid = int(pids[0])
    try:
        os.kill(pid, 0)
    except PermissionError:
        pass  # process exists but owned by another user
    except ProcessLookupError:
        print(f"Error: found PID {pid} but it is not running", file=sys.stderr)
        sys.exit(1)
    return pid


def _read_proc_stat(pid: int) -> dict[str, int] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        after_comm = text.rsplit(")", 1)[1].strip().split()
        return {
            "ppid": int(after_comm[1]),
            "utime": int(after_comm[11]),
            "stime": int(after_comm[12]),
            "num_threads": int(after_comm[17]),
        }
    except (IndexError, ValueError):
        return None


def discover_process_tree(root_pid: int) -> set[int]:
    children_by_parent: dict[int, list[int]] = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        stat = _read_proc_stat(pid)
        if stat is None:
            continue
        children_by_parent.setdefault(stat["ppid"], []).append(pid)

    seen: set[int] = set()
    queue: deque[int] = deque([root_pid])
    while queue:
        pid = queue.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(children_by_parent.get(pid, []))
    return seen


def _read_status_memory(pid: int) -> tuple[int, int]:
    rss = 0
    vsz = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
            elif line.startswith("VmSize:"):
                vsz = int(line.split()[1])
    except OSError:
        pass
    return rss, vsz


def _read_smaps_rollup(pid: int) -> tuple[int | None, int | None, str]:
    pss = None
    private_kb = 0
    saw_private = False
    try:
        lines = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
    except PermissionError:
        return None, None, f"cannot_read_smaps_rollup_pid_{pid}"
    except OSError:
        return None, None, ""

    for line in lines:
        if line.startswith("Pss:"):
            pss = int(line.split()[1])
        elif line.startswith(("Private_Clean:", "Private_Dirty:", "Private_Hugetlb:")):
            private_kb += int(line.split()[1])
            saw_private = True
    return pss, private_kb if saw_private else None, ""


def _read_io(pid: int) -> dict[str, int]:
    values = {"rchar": 0, "wchar": 0, "read_bytes": 0, "write_bytes": 0}
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in values:
                values[key] = int(value.strip())
    except OSError:
        pass
    return values


def _query_gpu(process_pids: set[int], has_nvidia_smi: bool) -> dict[str, float | int]:
    metrics = {
        "server_vram_mib": 0,
        "total_gpu_mem_mib": 0,
        "total_gpu_used_mib": 0,
        "gpu_util_pct": 0.0,
    }
    if not has_nvidia_smi:
        return metrics

    vram_out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    pid_strings = {str(pid) for pid in process_pids}
    for line in vram_out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) == 2 and cols[0] in pid_strings:
            try:
                metrics["server_vram_mib"] += int(cols[1])
            except ValueError:
                pass

    gpu_out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    util_vals = []
    for line in gpu_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            metrics["total_gpu_mem_mib"] += int(parts[0])
            metrics["total_gpu_used_mib"] += int(parts[1])
            util_vals.append(float(parts[2]))
        except ValueError:
            pass
    metrics["gpu_util_pct"] = round(sum(util_vals) / len(util_vals), 1) if util_vals else 0.0
    return metrics


def _observed_hardware_profile(row: dict[str, Any]) -> str:
    if int(row.get("server_vram_mib") or 0) > 0:
        return "cpu_gpu"
    if int(row.get("total_gpu_mem_mib") or 0) > 0:
        return "cpu_only_with_gpu_visible"
    return "cpu_only"


def _empty_resource_row(root_pid: int) -> dict[str, Any]:
    return {field: "" for field in RESOURCE_FIELDNAMES} | {"root_pid": root_pid}


def _sample_process_tree(root_pid: int, previous_pids: set[int], has_nvidia_smi: bool) -> tuple[dict[str, Any], set[int]]:
    pids = discover_process_tree(root_pid)
    row = _empty_resource_row(root_pid)
    row["process_count"] = len(pids)
    row["new_pids"] = len(pids - previous_pids)
    row["gone_pids"] = len(previous_pids - pids)

    pss_known = True
    uss_known = True
    pss_total = 0
    uss_total = 0
    warnings = []
    cpu_user_ticks = 0
    cpu_system_ticks = 0

    for pid in pids:
        rss, vsz = _read_status_memory(pid)
        row["rss_kb"] = int(row.get("rss_kb") or 0) + rss
        row["vsz_kb"] = int(row.get("vsz_kb") or 0) + vsz

        pss, uss, warning = _read_smaps_rollup(pid)
        if warning:
            warnings.append(warning)
        if pss is None:
            pss_known = False
        else:
            pss_total += pss
        if uss is None:
            uss_known = False
        else:
            uss_total += uss

        stat = _read_proc_stat(pid)
        if stat:
            cpu_user_ticks += stat["utime"]
            cpu_system_ticks += stat["stime"]
            row["thread_count"] = int(row.get("thread_count") or 0) + stat["num_threads"]

        io_values = _read_io(pid)
        row["io_rchar_bytes"] = int(row.get("io_rchar_bytes") or 0) + io_values["rchar"]
        row["io_wchar_bytes"] = int(row.get("io_wchar_bytes") or 0) + io_values["wchar"]
        row["io_read_bytes"] = int(row.get("io_read_bytes") or 0) + io_values["read_bytes"]
        row["io_write_bytes"] = int(row.get("io_write_bytes") or 0) + io_values["write_bytes"]

    row["pss_kb"] = pss_total if pss_known else ""
    row["uss_kb"] = uss_total if uss_known else ""
    row["cpu_user_s"] = round(cpu_user_ticks / CLOCK_TICKS, 6)
    row["cpu_system_s"] = round(cpu_system_ticks / CLOCK_TICKS, 6)
    row["cpu_total_s"] = round((cpu_user_ticks + cpu_system_ticks) / CLOCK_TICKS, 6)
    row |= _query_gpu(pids, has_nvidia_smi)
    row["observed_hardware_profile"] = _observed_hardware_profile(row)
    row["sampling_warning"] = ";".join(sorted(set(warnings)))
    return row, pids


def resource_monitor(
    pid: int,
    resource_csv: Path,
    stop_event: threading.Event,
    sample_interval: float,
) -> None:
    start = time.monotonic()
    previous_row: dict[str, Any] | None = None
    previous_pids: set[int] = set()
    has_nvidia_smi = subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0

    with resource_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESOURCE_FIELDNAMES)
        writer.writeheader()

        while not stop_event.is_set():
            try:
                now = time.monotonic()
                row, current_pids = _sample_process_tree(pid, previous_pids, has_nvidia_smi)
                row["ts_epoch"] = int(time.time())
                row["elapsed_s"] = round(now - start, 3)
                row["sample_dt_s"] = "" if previous_row is None else round(float(row["elapsed_s"]) - float(previous_row["elapsed_s"]), 6)

                if previous_row is not None and row["sample_dt_s"]:
                    dt = float(row["sample_dt_s"])
                    cpu_delta = float(row["cpu_total_s"] or 0) - float(previous_row["cpu_total_s"] or 0)
                    row["cpu_pct"] = round((cpu_delta / dt) * 100, 2) if dt > 0 else ""
                    for key, rate_key in [
                        ("io_rchar_bytes", "io_rchar_bps"),
                        ("io_wchar_bytes", "io_wchar_bps"),
                        ("io_read_bytes", "io_read_bps"),
                        ("io_write_bytes", "io_write_bps"),
                    ]:
                        delta = int(row[key] or 0) - int(previous_row[key] or 0)
                        row[rate_key] = round(delta / dt, 2) if dt > 0 else ""

                writer.writerow(row)
                f.flush()
                previous_row = row
                previous_pids = current_pids
            except Exception as e:
                print(f"Warning: resource_monitor sampling error: {e}", file=sys.stderr)

            stop_event.wait(sample_interval)


def get_audio_duration(audio: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _extract_response_payload(resp_out: Path, mode: str) -> dict[str, Any] | None:
    if not resp_out.exists():
        return None
    text = resp_out.read_text(errors="replace")
    if mode == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    payload = None
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "transcript.text.done":
            done_text = event.get("text", "")
            try:
                payload = json.loads(done_text)
            except json.JSONDecodeError:
                payload = None
    return payload


def _extract_transcript_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("transcript"), str):
        return payload["transcript"]
    segments = payload.get("segments") or payload.get("lines") or []
    if isinstance(segments, list):
        return " ".join(str(seg.get("text", "")) for seg in segments if isinstance(seg, dict)).strip()
    return ""


def normalize_spanish_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    translation = str.maketrans({ch: " " for ch in string.punctuation + "¿¡“”‘’«»…—–"})
    text = text.translate(translation)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(reference_transcript: Path | None, payload: dict[str, Any] | None) -> float | None:
    if reference_transcript is None:
        return None
    from jiwer import wer

    reference = normalize_spanish_text(reference_transcript.read_text(errors="replace"))
    hypothesis = normalize_spanish_text(_extract_transcript_text(payload))
    return float(wer(reference, hypothesis))


def _payload_to_diarization_turns(payload: dict[str, Any] | None) -> list[tuple[float, float, str]]:
    if not payload:
        return []
    turns = []
    diarization = payload.get("diarization")
    if isinstance(diarization, list) and diarization:
        source = diarization
    else:
        source = payload.get("segments") or payload.get("lines") or []
    for item in source:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        if speaker in (None, "", -1, "-1"):
            continue
        try:
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if end > start:
            turns.append((start, end, str(speaker)))
    return turns


def compute_der(
    reference_diarization: Path | None,
    payload: dict[str, Any] | None,
    collar: float,
    skip_overlap: bool,
) -> float | None:
    if reference_diarization is None:
        return None
    from pyannote.core import Annotation, Segment
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    references_by_uri = load_rttm(reference_diarization)
    if not references_by_uri:
        return None
    reference = next(iter(references_by_uri.values()))

    hypothesis = Annotation(uri=reference.uri)
    for idx, (start, end, speaker) in enumerate(_payload_to_diarization_turns(payload)):
        hypothesis[Segment(start, end), f"turn_{idx}"] = speaker

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    return float(metric(reference, hypothesis))


def backfill_resource_csv(resource_csv: Path, values: dict[str, Any]) -> None:
    if not resource_csv.exists():
        return
    with resource_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in values.items():
            row[key] = "" if value is None else value
    with resource_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESOURCE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def classify_resource_csv(resource_csv: Path) -> str:
    if not resource_csv.exists():
        return ""
    saw_gpu = False
    saw_server_gpu = False
    with resource_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            saw_gpu = saw_gpu or int(float(row.get("total_gpu_mem_mib") or 0)) > 0
            saw_server_gpu = saw_server_gpu or int(float(row.get("server_vram_mib") or 0)) > 0
    if saw_server_gpu:
        return "cpu_gpu"
    if saw_gpu:
        return "cpu_only_with_gpu_visible"
    return "cpu_only"


def run_rep(
    rep: int,
    audio: Path,
    mode: str,
    url: str,
    pid: int,
    out_dir: Path,
    sample_interval: float,
    audio_seconds: float | None,
    reference_transcript: Path | None,
    reference_diarization: Path | None,
    der_collar: float,
    der_skip_overlap: bool,
) -> dict[str, Any]:
    resource_csv = out_dir / f"resource_{rep}.csv"
    resp_out = out_dir / f"response_{rep}.{mode}"
    curl_metrics = out_dir / f"curl_metrics_{rep}.txt"
    time_metrics = out_dir / f"time_metrics_{rep}.txt"

    print(f"\n=== Repetition {rep} ===")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=resource_monitor,
        args=(pid, resource_csv, stop_event, sample_interval),
        daemon=True,
    )
    monitor_thread.start()

    curl_write_out = (
        "time_namelookup=%{time_namelookup}\n"
        "time_connect=%{time_connect}\n"
        "time_starttransfer=%{time_starttransfer}\n"
        "time_total=%{time_total}\n"
        "http_code=%{http_code}\n"
    )

    curl_cmd = [
        "/usr/bin/time", "-v", "-o", str(time_metrics),
        "curl", "-sS",
        "-X", "POST", url,
        "-F", f"file=@{audio}",
        "-F", "model=whisper-1",
        "-w", curl_write_out,
        "-o", str(resp_out),
    ]
    if mode == "sse":
        curl_cmd.insert(5, "-N")
        curl_cmd.extend(["-F", "stream=true"])

    proc = None
    t_start = time.perf_counter_ns()
    try:
        with open(curl_metrics, "w") as curl_out:
            proc = subprocess.run(curl_cmd, stdout=curl_out)
    finally:
        t_end = time.perf_counter_ns()
        stop_event.set()
        monitor_thread.join()

    if proc is not None and proc.returncode != 0:
        print(
            f"Warning: curl exited with code {proc.returncode} on rep {rep}",
            file=sys.stderr,
        )

    wall_seconds = (t_end - t_start) / 1_000_000_000
    throughput = (audio_seconds / wall_seconds) if audio_seconds and wall_seconds > 0 else None
    payload = _extract_response_payload(resp_out, mode)
    wer_value = compute_wer(reference_transcript, payload)
    der_value = compute_der(reference_diarization, payload, der_collar, der_skip_overlap)
    observed_hardware_profile = classify_resource_csv(resource_csv)

    backfill_resource_csv(
        resource_csv,
        {
            "observed_hardware_profile": observed_hardware_profile,
            "audio_seconds": round(audio_seconds, 6) if audio_seconds is not None else None,
            "wall_seconds": round(wall_seconds, 6),
            "transcription_throughput": round(throughput, 6) if throughput is not None else None,
            "wer": round(wer_value, 6) if wer_value is not None else None,
            "der": round(der_value, 6) if der_value is not None else None,
            "der_collar_s": der_collar if reference_diarization else None,
            "der_skip_overlap": der_skip_overlap if reference_diarization else None,
            "wer_normalization": "spanish-friendly" if reference_transcript else None,
        },
    )

    print(f"wall_seconds_rep{rep}={wall_seconds:.3f}")
    if throughput is not None:
        print(f"transcription_throughput_rep{rep}={throughput:.3f}")
    if wer_value is not None:
        print(f"wer_rep{rep}={wer_value:.4f}")
    if der_value is not None:
        print(f"der_rep{rep}={der_value:.4f}")

    if mode == "sse" and resp_out.exists():
        text = resp_out.read_text(errors="replace")
        print(f"[sse event counts rep {rep}]")
        print(f"progress_events={text.count('transcript.progress')}")
        print(f"delta_events={text.count('transcript.text.delta')}")
        print(f"done_events={text.count('transcript.text.done')}")
        print(f"done_markers={text.count('[DONE]')}")

    return {
        "wall_seconds": wall_seconds,
        "throughput": throughput,
        "wer": wer_value,
        "der": der_value,
        "observed_hardware_profile": observed_hardware_profile,
    }


def compute_peaks(resource_csv: Path) -> dict[str, Any]:
    peaks: dict[str, Any] = {
        "peak_rss_kb": 0,
        "peak_pss_kb": "",
        "peak_uss_kb": "",
        "peak_vsz_kb": 0,
        "peak_cpu_pct": 0.0,
        "peak_io_rchar_bps": 0.0,
        "peak_io_wchar_bps": 0.0,
        "peak_io_read_bps": 0.0,
        "peak_io_write_bps": 0.0,
        "peak_server_vram_mib": 0,
        "peak_total_gpu_used_mib": 0,
        "peak_gpu_util_pct": 0.0,
    }
    if not resource_csv.exists():
        return peaks

    pss_values = []
    uss_values = []
    with resource_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            peaks["peak_rss_kb"] = max(peaks["peak_rss_kb"], int(float(row.get("rss_kb") or 0)))
            peaks["peak_vsz_kb"] = max(peaks["peak_vsz_kb"], int(float(row.get("vsz_kb") or 0)))
            for key in ["cpu_pct", "io_rchar_bps", "io_wchar_bps", "io_read_bps", "io_write_bps", "gpu_util_pct"]:
                peaks[f"peak_{key}"] = max(float(peaks[f"peak_{key}"]), float(row.get(key) or 0))
            peaks["peak_server_vram_mib"] = max(peaks["peak_server_vram_mib"], int(float(row.get("server_vram_mib") or 0)))
            peaks["peak_total_gpu_used_mib"] = max(peaks["peak_total_gpu_used_mib"], int(float(row.get("total_gpu_used_mib") or 0)))
            if row.get("pss_kb") not in (None, ""):
                pss_values.append(int(float(row["pss_kb"])))
            if row.get("uss_kb") not in (None, ""):
                uss_values.append(int(float(row["uss_kb"])))
    peaks["peak_pss_kb"] = max(pss_values) if pss_values else ""
    peaks["peak_uss_kb"] = max(uss_values) if uss_values else ""
    return peaks


def main() -> None:
    args = parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"Error: audio file not found: {audio}", file=sys.stderr)
        sys.exit(1)
    if args.reference_transcript and not args.reference_transcript.is_file():
        print(f"Error: reference transcript not found: {args.reference_transcript}", file=sys.stderr)
        sys.exit(1)
    if args.reference_diarization and not args.reference_diarization.is_file():
        print(f"Error: reference RTTM not found: {args.reference_diarization}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pid = find_server_pid(args.server_pid, args.server_match)
    audio_seconds = get_audio_duration(audio)

    print(f"server_pid={pid}")
    print(f"audio={audio}")
    print(f"audio_seconds={audio_seconds if audio_seconds is not None else ''}")
    print(f"mode={args.mode}")
    print(f"reps={args.reps}")
    print(f"url={args.url}")
    print(f"out_dir={out_dir}")

    rep_results: list[dict[str, Any]] = []
    for rep in range(1, args.reps + 1):
        rep_results.append(
            run_rep(
                rep,
                audio,
                args.mode,
                args.url,
                pid,
                out_dir,
                args.sample_interval,
                audio_seconds,
                args.reference_transcript,
                args.reference_diarization,
                args.der_collar,
                args.der_skip_overlap,
            )
        )

    summary_lines: list[str] = []
    summary_lines.append(f"audio={audio}")
    summary_lines.append(f"audio_seconds={audio_seconds if audio_seconds is not None else ''}")
    summary_lines.append(f"mode={args.mode}")
    summary_lines.append(f"reps={args.reps}")
    summary_lines.append(f"url={args.url}")
    summary_lines.append(f"server_pid={pid}")
    summary_lines.append(f"sample_interval={args.sample_interval}")
    summary_lines.append(f"reference_transcript={args.reference_transcript or ''}")
    summary_lines.append(f"reference_diarization={args.reference_diarization or ''}")
    summary_lines.append(f"der_collar_s={args.der_collar if args.reference_diarization else ''}")
    summary_lines.append(f"der_skip_overlap={args.der_skip_overlap if args.reference_diarization else ''}")
    summary_lines.append(f"wer_normalization={'spanish-friendly' if args.reference_transcript else ''}")
    summary_lines.append(f"wall_total_seconds={sum(float(r['wall_seconds']) for r in rep_results):.3f}")
    summary_lines.append("")

    for rep in range(1, args.reps + 1):
        result = rep_results[rep - 1]
        summary_lines.append(f"wall_seconds_rep{rep}={result['wall_seconds']:.3f}")
        if result["throughput"] is not None:
            summary_lines.append(f"transcription_throughput_rep{rep}={result['throughput']:.6f}")
        if result["wer"] is not None:
            summary_lines.append(f"wer_rep{rep}={result['wer']:.6f}")
        if result["der"] is not None:
            summary_lines.append(f"der_rep{rep}={result['der']:.6f}")
        summary_lines.append(f"[rep {rep} peaks]")
        peaks = compute_peaks(out_dir / f"resource_{rep}.csv")
        for k, v in peaks.items():
            summary_lines.append(f"{k}={v}")
        summary_lines.append("")
        curl_metrics = out_dir / f"curl_metrics_{rep}.txt"
        if curl_metrics.exists():
            summary_lines.append(f"[curl rep {rep}]")
            summary_lines.append(curl_metrics.read_text())
        time_metrics = out_dir / f"time_metrics_{rep}.txt"
        if time_metrics.exists():
            summary_lines.append(f"[time rep {rep}]")
            summary_lines.append(time_metrics.read_text())
        summary_lines.append("")

    summary_lines.append("[outputs]")
    for rep in range(1, args.reps + 1):
        summary_lines.append(f"resource_csv_rep{rep}={out_dir}/resource_{rep}.csv")
        summary_lines.append(f"response_rep{rep}={out_dir}/response_{rep}.{args.mode}")
        summary_lines.append(f"curl_metrics_rep{rep}={out_dir}/curl_metrics_{rep}.txt")
        summary_lines.append(f"time_metrics_rep{rep}={out_dir}/time_metrics_{rep}.txt")

    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
