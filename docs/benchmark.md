# ASR + diarization benchmark leaderboard

Backend/model leaderboard for the full **transcription + diarization** pipeline,
produced with `coro-bench`. Use it to pick a backend; reproduce it on your
own data before trusting absolute numbers.

> **Read the caveats.** These runs are a **small AMI English sample** on one
> laptop GPU. They are a *relative* signal, not an absolute quality verdict —
> reproduce on data that matches your domain/language (the bench ships loaders
> for AMI, VoxConverse and Common Voice; see *Reproduce* below).

## Hardware & setup

- GPU: **RTX 3070 Laptop (8 GB)**, CPU: loaded laptop CPU.
- Diarization: **NeMo Sortformer** (`nvidia/diar_streaming_sortformer_4spk-v2`).
- Pipeline: `full-memory`. ASR precision: **fp16** on GPU, **int8** on CPU.
- `--reps 2`; quality scored from rep 1, performance averaged across reps.

## Metrics

- **ORC-WER** — speaker-agnostic word error (meeteval greedy ORC-WER); *norm* =
  punctuation stripped + whitespace collapsed. **Lower is better.** This is the
  headline ASR-quality number.
- **DER** — Diarization Error Rate (meeteval `md_eval_22`, collar 0). **Lower is
  better.** Only meaningful when diarization is on and the reference is
  multi-speaker.
- **RTFx** — audio seconds ÷ processing wall seconds. **Higher is faster**
  (10× = 10 s of audio per 1 s of compute).
- **Peak VRAM / host RAM** — peak resident during inference (per server process).

## Leaderboard — GPU (diarization on)

Sample: **5 AMI clips** (`ES2004a`, `IB4001`, `IN1001`, `IS1009a`, `TS3003a`;
60 s each, 300 s total). Sorted by quality (best first).

| Backend / model | norm ORC-WER ↓ | DER ↓ | RTFx (GPU) ↑ | Peak VRAM | Peak host RAM |
|---|---:|---:|---:|---:|---:|
| **faster-whisper `large-v3-turbo`** | **0.398** | **0.278** | 18.5× | ~3.0 GB | ~3.0 GB |
| onnx-asr `parakeet-tdt-0.6b-v3` | 0.439 | 0.352 | 6.7× | ~0.8 GB † | ~5.5 GB |
| faster-whisper `small` | 0.452 | 0.371 | **31.1×** | ~1.5 GB | ~3.0 GB |
| faster-whisper `medium` | 0.496 | 0.372 | 16.0× | ~2.8 GB | ~2.9 GB |

† Parakeet VRAM is per-process and **under-reports** onnxruntime's CUDA
allocations; treat it as a lower bound. Its lower RTFx and higher host RAM here
reflect onnx-asr's per-request overhead on short clips — its offline-batched
throughput on long audio is much higher.

**Highlights**
- **`large-v3-turbo` wins on both WER and DER**, fits in ~3 GB VRAM, is
  multilingual, and is the most *robust* — it held up on the hard `IN1001` clip
  where `small`/`medium` collapsed (~0.98 norm ORC-WER).
- `medium` scored *worse* than `small` on this sample — driven by the
  pathological clips; another reason not to over-read a small sample.

## Leaderboard — CPU (ASR only, diarization off)

Sample: **2 AMI clips** (`IB4001`, `IN1001`; 120 s). int8. Diarization off, so
ORC-WER is still valid but DER/cpWER are not reported (Sortformer on CPU is slow;
run it on GPU). WER here is **not** comparable to the 5-clip GPU table above.

| Backend / model (int8, CPU) | norm ORC-WER ↓ | RTFx (CPU) ↑ | Peak host RAM |
|---|---:|---:|---:|
| **onnx-asr `parakeet-tdt-0.6b-v3`** | **0.424** | **8.2×** | ~2.1 GB |
| faster-whisper `small` | 0.576 | 2.8× | ~1.5 GB |

**Parakeet is the CPU pick** — faster *and* more accurate than faster-whisper
small on CPU.

## Other backends

- **onnx-asr `whisper-*`** — *not recommended.* onnx-asr decodes Whisper as
  VAD-chunked 30 s windows (greedy, no long-form context / temperature
  fallback). On `IB4001` it scored ~0.43 norm ORC-WER (vs faster-whisper small's
  0.31) and ran at **~0.33× RT** even on GPU. Use **faster-whisper** for Whisper
  models. (onnx-asr's strength is Parakeet.)
- **onnx-genai `nemotron-…` (streaming)** — a cache-aware **streaming** model for
  low-latency real-time use, not batch throughput (~10× GPU, impractical on CPU).
  Not re-run in this matrix; see the README *Benchmarks* section.

## Suggestions for the end user

- **Best overall (GPU):** **faster-whisper `large-v3-turbo`** — best WER + DER,
  multilingual, ~3 GB VRAM, comfortably fits an 8 GB GPU. Recommended default.
- **Max GPU throughput:** faster-whisper `small` (~31× RTFx) when speed matters
  more than the last few WER points.
- **CPU deployment:** onnx-asr `parakeet-tdt-0.6b-v3` (int8) — fastest and most
  accurate on CPU.
- **Lowest VRAM:** parakeet (but watch host RAM) or faster-whisper `small`.
- **Real-time / streaming:** onnx-genai `nemotron` (cache-aware streaming).
- **Avoid:** running Whisper through the onnx-asr backend — slower and less
  accurate than faster-whisper.

## Reproduce

```bash
# 1) Build clips (gold AMI references):
python -m coro.bench.utils.make_ami_clip IB4001 \
  --ami-root ./amicorpus --start 180 --duration 60 --out-dir clips

# 2) Start a server (pick the backend/model/device), wait for /health ready:
CORO_BACKEND_ASR=faster-whisper CORO_MODEL_ASR=openai/whisper-large-v3-turbo \
CORO_ASR_DEVICE=cuda CORO_ASR_COMPUTE_TYPE=float16 \
CORO_BACKEND_DIARIZATION=nemo CORO_PIPELINE=full-memory \
  coro --port 8123

# 3) Run the full benchmark (quality + performance):
coro-bench all --clips-dir clips --server-url http://127.0.0.1:8123 \
  --server-pid <PID> --reps 2 --out-dir run

# 4) (optional) side-by-side ref/hyp alignment viz:
python -m coro.bench.utils.visualize_quality run --alignment tcp cp
```

Other dataset loaders: `make_rttm_clip` (VoxConverse / diarization-only DER),
`make_common_voice_clips` (Common Voice WER). A trustworthy **Spanish** DER+WER
target (Albayzín-RTVE2020) is gated behind an RTVE licence — see the README
*Benchmark datasets* note; apply for access if you need Spanish numbers.
