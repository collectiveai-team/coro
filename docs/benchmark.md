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
- Diarization: **NeMo Sortformer** (`nvidia/diar_streaming_sortformer_4spk-v2`,
  CC-BY-4.0 — see the README *Model licensing* table).
- Pipeline: `full-memory`. ASR precision: **fp16** on GPU, **fp32** on CPU.
- `--reps 2`; quality scored from rep 1, performance averaged across reps.

> **RTFx numbers are only comparable within a table.** Every table below is
> labelled with its clip length, diarization state and device, because those
> three conditions move RTFx by more than an order of magnitude — see
> [Reading the throughput tables](#reading-the-throughput-tables).

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

## Reading the throughput tables

RTFx is a property of a *measurement*, not of a model. Three conditions change
it by more than an order of magnitude, and every table in this repo now names
all three:

| Condition | Effect on RTFx |
|---|---|
| **Clip length** | Per-request fixed cost (upload, decode, model entry) is amortised over the audio. On 60 s clips it dominates; on 600 s clips it is negligible. This alone explains most of the gap between the 60 s GPU table below and the long-form README table. |
| **Diarization on/off** | RTFx is measured end-to-end over the **Transcription Pipeline**, so a diarization-on number includes Sortformer. Measured cost of turning it on: **5.03× → 3.84×** (−24%) for the default backend on CPU. |
| **Device** | CPU vs GPU, and for `onnx-asr` also the execution provider. |

Two tables in this repo previously disagreed by ~18× on parakeet GPU RTFx
(README `~120×` vs the 6.7× below) with nothing recording why. They were not
measuring the same thing: the README figure is **long-form, ASR-only**, the
table below is **60 s clips with diarization on**. Neither figure is
reproducible from artifacts committed to this repo, so both are now labelled as
historical; the CPU tables below were re-measured for this document and their
artifacts are described in [Measured defaults](#measured-defaults-cpu).

## Leaderboard — GPU, 60 s clips, diarization on *(historical, not reproducible)*

Sample: **5 AMI clips** (`ES2004a`, `IB4001`, `IN1001`, `IS1009a`, `TS3003a`;
60 s each, 300 s total). Sorted by quality (best first). Short clips plus
diarization make these the *lowest* RTFx conditions in this document.

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

## Leaderboard — CPU, 60 s clips, diarization off *(historical, not reproducible)*

Sample: **2 AMI clips** (`IB4001`, `IN1001`; 120 s). int8. Diarization off, so
ORC-WER is still valid but DER/cpWER are not reported. WER here is **not**
comparable to the 5-clip GPU table above.

| Backend / model (int8, CPU) | norm ORC-WER ↓ | RTFx (CPU) ↑ | Peak host RAM |
|---|---:|---:|---:|
| **onnx-asr `parakeet-tdt-0.6b-v3`** | **0.424** | **8.2×** | ~2.1 GB |
| faster-whisper `small` | 0.576 | 2.8× | ~1.5 GB |

## Measured defaults (CPU)

The tables in this section back the current **Server Startup Selection**
defaults. Unlike the historical tables above, every number here comes from one
documented `coro-bench all` run per row.

**Conditions.** `full-memory` pipeline, `CORO_ASR_DEVICE=cpu`, diarization off
unless stated, `--reps 1`, one 600 s-clip **Workload Set** per lane. Server
VRAM delta was 0 on every run — these are **CPU-Only Runs** in substance even
where the harness labels the **Observed Hardware Profile** `cpu+gpu` (an
unrelated process was using the GPU). Absolute WER is inflated on both AMI
lanes: AMI `Mix-Headset` is hard far-field/overlap audio, and the normalizer
strips punctuation only (it does not case-fold or expand numbers). Compare
*within* a column.

### ASR Model Selection — English, AMI

4 × 600 s AMI clips (`IB4001` ×2, `IN1001` ×2; 40 min), diarization off.

| Backend / **ASR Model Selection** | norm ORC-WER ↓ | RTFx ↑ | Loaded PSS | Peak PSS |
|---|---:|---:|---:|---:|
| **onnx-asr `nemo-parakeet-tdt-0.6b-v3`** (default) | **0.5141** | **5.03×** | **2.73 GB** | **3.31 GB** |
| faster-whisper `openai/whisper-medium` | 0.5159 | 0.58× | 3.78 GB | 5.30 GB |

### ASR Model Selection — Spanish, FLEURS

60 × FLEURS `es_419` test utterances (11.9 min, CC-BY-4.0), single-speaker, so
WER only.

| Backend / **ASR Model Selection** | norm WER ↓ | RTFx ↑ | Loaded PSS | Peak PSS |
|---|---:|---:|---:|---:|
| **onnx-asr `nemo-parakeet-tdt-0.6b-v3`** (default) | **0.0579** | **5.06×** | **2.73 GB** | **2.89 GB** |
| faster-whisper `openai/whisper-medium` | 0.0757 | 0.57× | 3.75 GB | 4.07 GB |

**Parakeet is the CPU pick, and the Spanish pick.** English meeting quality is a
wash (−0.2 pp, within noise) but Spanish WER drops **23% relative** and
throughput is **~8.8× higher** at ~1 GB less resident memory. Measured Spanish
WER is above parakeet's published 3.45% FLEURS figure; the normalizer here does
not case-fold, and this measurement predates the **ASR Windowing** overlap fix.

### Quantization: int8 is a memory tool, not a speed tool

Same two lanes, default backend, `CORO_ASR_QUANTIZATION=int8` versus unset.

| Lane | precision | norm WER ↓ | RTFx ↑ | Loaded PSS |
|---|---|---:|---:|---:|
| AMI | fp32 *(default)* | **0.5141** | 5.03× | 2.73 GB |
| AMI | int8 | 0.5318 | 5.06× | **1.65 GB** |
| FLEURS es | fp32 *(default)* | **0.0579** | **5.06×** | 2.73 GB |
| FLEURS es | int8 | 0.0612 | 4.88× | **1.27 GB** |

int8 bought **no throughput** (+0.6% on AMI, −3.6% on Spanish — both inside
run-to-run noise) and cost **3–6% relative WER**, while saving **1.1–1.5 GB** of
resident memory. That is the expected shape for an encoder-heavy transducer:
it is compute-bound, so `DynamicQuantizeLinear` overhead is pure loss, and the
reference host has no VNNI to recover it. Set `CORO_ASR_QUANTIZATION=int8` only
to fit a memory budget, never to go faster.

### Diarization Model Selection: Sortformer v2.1 was **not** adopted

A/B over 8 × 600 s AMI clips (80 min, 6 meetings), **ASR Model Selection held
constant**, `CORO_DIARIZATION_DEVICE=cpu`.

| **Diarization Model Selection** | DER ↓ | cpWER ↓ | norm cpWER ↓ | RTFx ↑ |
|---|---:|---:|---:|---:|
| **`nvidia/diar_streaming_sortformer_4spk-v2`** (default) | **0.6218** | **0.6573** | **0.5575** | **3.6–3.8×** |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | 0.6237 | 0.6599 | 0.5623 | 3.1× |

v2.1's model card reports −38% relative DER on AMI SDM. **That did not
reproduce here.** v2.1 won 5 of 8 clips and lost 3, the pooled result is
marginally *worse* on all three quality metrics, and it ran ~15% slower. The
default therefore stays on v2.

The likely reason is measurement sensitivity, not a claim that v2.1 is a bad
model: in this pipeline the hypothesis speaker turns are derived from ASR
segment spans, so missed-speech (11.24%) and false-alarm (44.19%) are
*identical* across both arms and only the speaker-error term moves (6.75% v2 vs
6.94% v2.1). DER here is dominated by segmentation, which no diarization model
can change. Re-run this A/B after word-level speaker assignment and Sortformer
post-processing tuning land, when DER can actually discriminate.

### Diarization by default: stays off

Decided explicitly rather than by omission. `CORO_BACKEND_DIARIZATION` remains
`none`, so the out-of-the-box server is an **ASR-Only Server**. Measured cost of
switching it on for the default backend on CPU: **5.03× → 3.84× RTFx** (−24%)
and roughly **+1 GB** peak **Process-Tree PSS**, plus a ~500 MB model download
on first start. Against that, streaming Sortformer caps at 4 speakers — a
default that silently mis-attributes 5-speaker audio is worse than a default
that does not claim to attribute at all — and the DER this pipeline currently
produces (0.62) is not a result worth shipping as the default experience.
Revisit once speaker attribution improves. Turning it on is one setting:
`CORO_BACKEND_DIARIZATION=nemo`.

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

- **Default (and the CPU / Spanish pick):** onnx-asr
  `nemo-parakeet-tdt-0.6b-v3` at **fp32** — ~8.8× the throughput of
  `whisper-medium` on CPU, 23% lower Spanish WER, ~1 GB less resident memory.
  Nothing to configure.
- **Best English meeting quality (GPU):** faster-whisper `large-v3-turbo` — best
  WER + DER in the 60 s GPU table, multilingual, ~3 GB VRAM.
- **Max GPU throughput:** faster-whisper `small` (~31× RTFx) when speed matters
  more than the last few WER points.
- **Tight memory budget:** add `CORO_ASR_QUANTIZATION=int8` — it saves
  1.1–1.5 GB but does **not** make the default backend faster and costs 3–6%
  relative WER.
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

### Reproducing *Measured defaults (CPU)*

Same four steps, with these conditions per row:

- **AMI lane** — 8 × 600 s clips via `make_ami_clip` at
  `IB4001@0`, `IB4001@900`, `IN1001@0`, `IN1001@1200`, `ES2002b@300`,
  `ES2004a@300`, `IS1009a@300`, `TS3003a@300`. The ASR and quantization tables
  use the first four; the diarization A/B uses all eight.
- **Spanish lane** — 60 utterances from the FLEURS `es_419` **test** split
  (CC-BY-4.0), each transcoded to 16 kHz mono WAV with its
  `raw_transcription` as a single-speaker reference STM — the same
  `(<stem>.wav, <stem>.ref.stm)` shape `make_common_voice_clips` produces.
- **Server** — `CORO_ASR_DEVICE=cpu`, `CORO_PIPELINE=full-memory`,
  `CORO_BACKEND_DIARIZATION=none` except in the diarization A/B, where it is
  `nemo` with `CORO_DIARIZATION_DEVICE=cpu` and `CORO_MODEL_DIARIZATION` set to
  the arm under test. `--reps 1`.
- **Loaded PSS** is the harness's `baseline_pss_kb` (server with models
  resident, before the request); **Peak PSS** is `peak_pss_kb`.
