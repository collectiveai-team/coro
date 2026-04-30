#!/usr/bin/env bash
set -euo pipefail

AUDIO="${1:-}"
MODE="${2:-json}" # json or sse
URL="${URL:-http://localhost:8000/v1/audio/transcriptions}"
SERVER_MATCH="${SERVER_MATCH:-custom_server.py}"
OUT_DIR="${OUT_DIR:-/tmp/asr-bench-$(date +%Y%m%d-%H%M%S)}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"

if [[ -z "$AUDIO" || ! -f "$AUDIO" ]]; then
  echo "Usage: $0 /path/to/audio.wav [json|sse]"
  echo
  echo "Optional env:"
  echo "  URL=http://localhost:8000/v1/audio/transcriptions"
  echo "  SERVER_PID=12345"
  echo "  SERVER_MATCH=custom_server.py"
  echo "  OUT_DIR=/tmp/asr-bench"
  echo "  SAMPLE_INTERVAL=1"
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ -n "${SERVER_PID:-}" ]]; then
  PID="$SERVER_PID"
else
  PID="$(pgrep -f "$SERVER_MATCH" | head -n1 || true)"
fi

if [[ -z "$PID" ]]; then
  echo "Could not find server PID. Set SERVER_PID=..."
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Server PID $PID is not running"
  exit 1
fi

MEM_CSV="$OUT_DIR/memory.csv"
RESP_OUT="$OUT_DIR/response.${MODE}"
CURL_METRICS="$OUT_DIR/curl_metrics.txt"
TIME_METRICS="$OUT_DIR/time_metrics.txt"

echo "server_pid=$PID"
echo "audio=$AUDIO"
echo "mode=$MODE"
echo "out_dir=$OUT_DIR"

echo "ts_epoch,elapsed_s,rss_kb,vsz_kb,server_vram_mib,total_gpu_mem_mib,total_gpu_used_mib,gpu_util_pct" > "$MEM_CSV"

monitor() {
  local start
  start="$(date +%s)"

  while true; do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi

    local now elapsed rss vsz server_vram total_gpu_mem total_gpu_used gpu_util
    now="$(date +%s)"
    elapsed="$((now - start))"

    read -r rss vsz < <(ps -o rss=,vsz= -p "$PID" 2>/dev/null | awk '{print $1, $2}')
    rss="${rss:-0}"
    vsz="${vsz:-0}"

    server_vram="0"
    total_gpu_mem="0"
    total_gpu_used="0"
    gpu_util="0"

    if command -v nvidia-smi >/dev/null 2>&1; then
      server_vram="$(
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
          | awk -F, -v pid="$PID" '
              {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
                if ($1 == pid) sum += $2
              }
              END { print sum + 0 }
            '
      )"

      total_gpu_mem="$(
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
          | awk '{sum += $1} END {print sum + 0}'
      )"

      total_gpu_used="$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk '{sum += $1} END {print sum + 0}'
      )"

      gpu_util="$(
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
          | awk '{sum += $1; n += 1} END {if (n) printf "%.1f", sum / n; else print 0}'
      )"
    fi

    echo "$now,$elapsed,$rss,$vsz,$server_vram,$total_gpu_mem,$total_gpu_used,$gpu_util" >> "$MEM_CSV"
    sleep "$SAMPLE_INTERVAL"
  done
}

monitor &
MONITOR_PID="$!"

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

START_NS="$(date +%s%N)"

if [[ "$MODE" == "sse" ]]; then
  /usr/bin/time -v -o "$TIME_METRICS" \
    curl -N -sS \
      -X POST "$URL" \
      -F "file=@${AUDIO}" \
      -F "model=whisper-1" \
      -F "stream=true" \
      -w "time_namelookup=%{time_namelookup}\ntime_connect=%{time_connect}\ntime_starttransfer=%{time_starttransfer}\ntime_total=%{time_total}\nhttp_code=%{http_code}\n" \
      -o "$RESP_OUT" \
      > "$CURL_METRICS"
else
  /usr/bin/time -v -o "$TIME_METRICS" \
    curl -sS \
      -X POST "$URL" \
      -F "file=@${AUDIO}" \
      -F "model=whisper-1" \
      -w "time_namelookup=%{time_namelookup}\ntime_connect=%{time_connect}\ntime_starttransfer=%{time_starttransfer}\ntime_total=%{time_total}\nhttp_code=%{http_code}\n" \
      -o "$RESP_OUT" \
      > "$CURL_METRICS"
fi

END_NS="$(date +%s%N)"
cleanup
trap - EXIT

WALL_SECONDS="$(
  awk -v start="$START_NS" -v end="$END_NS" 'BEGIN { printf "%.3f", (end - start) / 1000000000 }'
)"

PEAKS="$(
  awk -F, '
    NR > 1 {
      if ($3 > peak_rss) peak_rss = $3
      if ($4 > peak_vsz) peak_vsz = $4
      if ($5 > peak_server_vram) peak_server_vram = $5
      if ($7 > peak_total_gpu_used) peak_total_gpu_used = $7
      if ($8 > peak_gpu_util) peak_gpu_util = $8
    }
    END {
      printf "peak_rss_kb=%s\npeak_vsz_kb=%s\npeak_server_vram_mib=%s\npeak_total_gpu_used_mib=%s\npeak_gpu_util_pct=%s\n", \
        peak_rss + 0, peak_vsz + 0, peak_server_vram + 0, peak_total_gpu_used + 0, peak_gpu_util + 0
    }
  ' "$MEM_CSV"
)"

SUMMARY="$OUT_DIR/summary.txt"

{
  echo "audio=$AUDIO"
  echo "mode=$MODE"
  echo "url=$URL"
  echo "server_pid=$PID"
  echo "wall_seconds=$WALL_SECONDS"
  echo
  echo "$PEAKS"
  echo
  echo "[curl]"
  cat "$CURL_METRICS"
  echo
  echo "[time]"
  cat "$TIME_METRICS"
  echo
  echo "[outputs]"
  echo "memory_csv=$MEM_CSV"
  echo "response=$RESP_OUT"
  echo "curl_metrics=$CURL_METRICS"
  echo "time_metrics=$TIME_METRICS"
} | tee "$SUMMARY"

if [[ "$MODE" == "sse" ]]; then
  echo
  echo "[sse event counts]"
  grep -c "transcript.progress" "$RESP_OUT" | awk '{print "progress_events=" $1}'
  grep -c "transcript.text.delta" "$RESP_OUT" | awk '{print "delta_events=" $1}'
  grep -c "transcript.text.done" "$RESP_OUT" | awk '{print "done_events=" $1}'
  grep -c "\\[DONE\\]" "$RESP_OUT" | awk '{print "done_markers=" $1}'
fi

echo
echo "Summary written to: $SUMMARY"