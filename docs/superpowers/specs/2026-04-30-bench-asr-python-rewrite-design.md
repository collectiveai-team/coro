# Design: bench_asr.py — Python rewrite of bench_asr.sh

Date: 2026-04-30

## Summary

Rewrite `tools/bench_asr.sh` as `tools/bench_asr.py`. Functionally equivalent but with cleaner
control flow, proper per-rep memory monitoring, and real argument parsing instead of positional-only
args and env-var-only options.

The bash script is deleted; the Python script replaces it entirely.

## CLI Interface

```
bench_asr.py <audio> [mode] [reps]
             [--url URL]
             [--server-pid PID]
             [--server-match PATTERN]
             [--out-dir PATH]
             [--sample-interval SECONDS]
```

| Argument | Default | Env var fallback |
|---|---|---|
| `audio` | required positional | — |
| `mode` | `json` | — |
| `reps` | `1` | — |
| `--url` | `http://localhost:8000/v1/audio/transcriptions` | `URL` |
| `--server-pid` | — | `SERVER_PID` |
| `--server-match` | `custom_server.py` | `SERVER_MATCH` |
| `--out-dir` | `/tmp/asr-bench-<timestamp>` | `OUT_DIR` |
| `--sample-interval` | `1` | `SAMPLE_INTERVAL` |

Env vars are fallbacks; explicit args take precedence. Implemented via `argparse` with
`default=os.environ.get(VAR, hardcoded_default)`.

## Architecture

Single file `tools/bench_asr.py`, no new dependencies (stdlib only: `argparse`, `csv`, `os`,
`subprocess`, `threading`, `time`).

### Functions

**`find_server_pid(server_pid, server_match) -> int`**
- If `server_pid` is provided, validate the process is running and return it.
- Otherwise use `subprocess` to run `pgrep -f <server_match>`, take first result.
- Exit with error message if no PID found or process not running.

**`memory_monitor(pid, mem_csv, stop_event, sample_interval)`**
- Runs in a `threading.Thread`.
- Writes CSV header on start.
- Loop: read RSS/VSZ from `ps`, optionally query `nvidia-smi` for VRAM/GPU util, append row, sleep.
- Exits when `stop_event.is_set()`.

**`run_rep(rep, audio, mode, url, pid, out_dir, sample_interval) -> float`**
- Computes per-rep file paths: `memory_N.csv`, `response_N.{mode}`, `curl_metrics_N.txt`, `time_metrics_N.txt`.
- Creates and starts monitor thread with its own `threading.Event`.
- Records start time with `time.perf_counter_ns()`.
- Runs `curl` via `subprocess.run` (same flags as bash version, including `/usr/bin/time -v`).
- Records end time, signals monitor stop event, joins thread.
- Prints per-rep wall time and SSE event counts (if mode=sse).
- Returns wall seconds as float.

**`compute_peaks(mem_csv) -> dict`**
- Reads a memory CSV, returns dict of peak values (rss, vsz, server_vram, total_gpu_used, gpu_util).

**`main()`**
- Parses args via `argparse`.
- Validates audio file exists.
- Calls `find_server_pid`.
- Creates `out_dir`.
- Prints run parameters.
- Loops over reps calling `run_rep`, collects wall times.
- Writes `summary.txt` with: parameters, per-rep wall time, per-rep peaks, curl/time file contents, output file list.
- Prints summary path.

## Output Files

Per rep:
- `memory_N.csv` — timestamp, elapsed_s, rss_kb, vsz_kb, server_vram_mib, total_gpu_mem_mib, total_gpu_used_mib, gpu_util_pct
- `response_N.{mode}` — raw curl response body
- `curl_metrics_N.txt` — curl `-w` timing output
- `time_metrics_N.txt` — `/usr/bin/time -v` output

Global:
- `summary.txt` — key=value parameters, per-rep wall times, per-rep peaks, contents of curl/time files, output file paths

## What is NOT changing

- Same CSV schema for memory files
- Same curl flags and `-w` format string
- Same `/usr/bin/time -v` invocation
- Same SSE event counting logic (grep equivalent via Python `str.count`)
- Same env var names as fallbacks (backwards compatible)
