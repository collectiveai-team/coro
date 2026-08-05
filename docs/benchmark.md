# ASR + diarization benchmark leaderboard

Backend/model leaderboard for the full **transcription + diarization** pipeline,
produced with `coro-bench`. Use it to pick a backend; reproduce it on your
own data before trusting absolute numbers.

> ### ⚠️ STALE — the four-backend comparison below predates four scoring fixes
>
> **Do not cite the 5-clip tables.** They were measured before:
>
> 1. **`e71dc71`** — ASR Windowing had no **Overlap Token Acceptance**, so every
>    window re-emitted its predecessor's final 2 s. These are 60 s clips, so each
>    one contains a seam. All backends over-emit; the inflation is not uniform
>    across them, so the *ordering* is not safe either.
> 2. **`c082427`** — the AMI reference builder silently dropped segments whose
>    annotation range ended on a `<disfmarker>`/`<gap>`/`<vocalsound>`. On this
>    sample 8% of reference words were missing (`IN1001` 70%, `TS3003a` 29%),
>    so correctly transcribed speech was scored as insertions.
> 3. **`7326916`** — clip references were sliced from rendered STM text, which
>    can only clamp a straddling segment's *times*; its words crossed the clip
>    edge intact, so clips were credited with speech their audio does not
>    contain.
> 4. **`f55c395`** — segments were not split at speaker changes, so a segment
>    spanning a turn attributed all its words to one speaker.
>
> The sample is also too small to carry its conclusions: **420 reference words
> across all five clips**, and `IN1001` has **3** (10 after the fix). A WER over
> a 3-word reference is noise, so the "most robust backend" claim below rests on
> nothing.
>
> **A properly sized measurement now exists for three of the four backends** —
> see *Current measurement* immediately below. It contradicts the 5-clip
> ordering rather than confirming it. Only faster-whisper `small` is still
> un-rerun; doing so is the outstanding work on this page.

## Current measurement (trustworthy)

Workload: **61 clips of 10 minutes** cut from the 30 meetings of the AMI **ES**
group — 10.2 h of audio, 88,515 reference words, materialized by
`coro.bench.utils.make_ami_clip_set`. This is the ADR 0008 measurement workload
and it is large enough to separate backends; the 5-clip sample below is not.

Configuration: `full-memory`, onnx-asr `nemo-parakeet-tdt-0.6b-v3` on CUDA
(fp32), NeMo `diar_streaming_sortformer_4spk-v2`.

Scored under the **Whisper English Text Schema** — the Whisper
`EnglishTextNormalizer` conventions the Open ASR Leaderboard uses, so these are
directly comparable with published numbers. A WER is meaningless without naming
its text schema; coro reports three, and they differ by more than a factor of
two on the same hypothesis.

The faster-whisper rows are fp16 (`CORO_ASR_COMPUTE_TYPE=float16`); parakeet is
fp32. Everything else is held fixed — same clips, same NeMo Sortformer
diarizer, same `full-memory` pipeline, and the default 2 word / 0.4 s
**Minimum Turn Threshold** in every run. All rows scored 61/61 clips with no
failures, no degenerate diarization, and the same 88,515-word reference.

| Backend / model | cpWER ↓ | ORC-WER ↓ | DI-cpWER ↓ | DER ↓ |
|---|---:|---:|---:|---:|
| onnx-asr `nemo-parakeet-tdt-0.6b-v3` | **0.1967** | **0.1742** | **0.1541** | **0.2831** |
| faster-whisper `large-v3-turbo` | 0.2170 | 0.1966 | 0.1871 | 0.3131 |
| faster-whisper `medium` | 0.2417 | 0.2267 | 0.2147 | 0.3342 |
| faster-whisper `small` | _not re-run_ | | | |

**Parakeet wins on every metric, reversing the 5-clip table below.** That table
ranked `large-v3-turbo` ahead of parakeet and recommended it as the default;
on a workload large enough to support a conclusion, parakeet leads
`large-v3-turbo` by 2.2 points of cpWER and 2.2 points of ORC-WER. Treat the
old ordering as an artefact of a 420-word sample and four scoring bugs, not as
a finding that was later overturned by a better model.

The ordering is also monotonic in model size here (`large-v3-turbo` >
`medium`), where the 5-clip sample had `medium` scoring *worse* than `small` —
another sign that sample was reading noise.

**DER varies across ASR backends**, which is easy to misread: the diarizer is
byte-identical in all three runs. DER is scored from the Hypothesis STM, whose
segment boundaries come from ASR segmentation, so a backend that segments
differently moves DER without the speaker timeline changing at all. DER here is
a property of the pipeline, not of Sortformer alone.

For scale: NVIDIA publish **11.31%** WER for parakeet on AMI **IHM**, which is
per-speaker headset audio, pre-segmented, with no diarization — a materially
easier task than the mixed-headset, diarized, unsegmented workload here. 17.42%
ORC-WER against that is a sane ratio.

Run cost, and why the wall times are not comparable: parakeet 9 min,
`large-v3-turbo` 33 min, `medium` 105 min. The `medium` run overlapped host
load averaging ~20 on a 16-thread box, and **host load is the dominant variable
in wall time on this hardware** — a run measured at load 41 elsewhere took 6×
its idle time. These wall times are therefore *not* an RTFx comparison. Quality
scores are unaffected: they are deterministic given fixed audio and a fixed
model. Peak whole-GPU memory (ASR + diarizer resident together) was ~3.3 GB for
`large-v3-turbo` and ~3.7 GB for `medium`, against parakeet's ~4.3 GB.

> **Read the caveats.** These runs are a **small AMI English sample** on one
> laptop GPU. They are a *relative* signal, not an absolute quality verdict —
> reproduce on data that matches your domain/language (the bench ships loaders
> for AMI, VoxConverse and Common Voice; see *Reproduce* below).

## Hardware & setup

- GPU: **RTX 3070 Laptop (8 GB)**, CPU: loaded laptop CPU.
- Diarization: **NeMo Sortformer** (`nvidia/diar_streaming_sortformer_4spk-v2`).
- Pipeline: `full-memory`. ASR precision: **fp16** on GPU, **int8** on CPU.
- `--reps 2`; quality scored from rep 1, performance averaged across reps.

The last two bullets describe the **voided 5-clip runs only**. The 61-clip
measurement used `coro-bench quality` (no `--reps`, no performance sampling),
fp16 for faster-whisper and fp32 for parakeet.

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

## Leaderboard — GPU (diarization on) — ⚠️ superseded, do not cite

Sample: **5 AMI clips** (`ES2004a`, `IB4001`, `IN1001`, `IS1009a`, `TS3003a`;
60 s each, 300 s total). Sorted by quality (best first). Retained only so the
re-run has something to compare against; every figure is void for the four
reasons above, and the sample could not support these conclusions even had the
scoring been correct.

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

**Highlights — both since disproven, retained as a record of the error**
- ~~`large-v3-turbo` wins on both WER and DER~~ — **wrong.** On the 61-clip
  workload parakeet leads it on cpWER, ORC-WER *and* DER. The win here came
  from a 420-word sample scored against a broken reference.
- ~~`medium` scored *worse* than `small`~~ — an inversion that does not survive
  a larger sample; the 61-clip ordering is monotonic in model size.

The one claim that still stands is the warning attached to them: do not
over-read a small sample. Both conclusions above were drawn from this table and
both were false.

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

Quality claims here come from the 61-clip measurement; throughput claims still
come from the voided 5-clip table, because no trustworthy RTFx numbers have
been collected yet. They are flagged individually below.

- **Best quality (GPU):** **onnx-asr `nemo-parakeet-tdt-0.6b-v3`** — best
  cpWER, ORC-WER and DER of the three backends measured on the 61-clip
  workload. **English only**, so it is not a candidate for multilingual
  deployments.
- **Best quality with multilingual support:** faster-whisper
  `large-v3-turbo` — 2.2 points of cpWER behind parakeet, ~3.3 GB peak GPU,
  comfortably fits an 8 GB card. Pick this when the audio is not English.
- **Max GPU throughput:** faster-whisper `small` — ⚠️ *based on the voided
  5-clip table (~31× RTFx); its quality has not been re-measured.*
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

### The ES clip workload set

For measurements that compare two runs — a baseline against a later
re-measurement — build the whole workload set in one reproducible command
instead of cutting clips meeting by meeting:

```bash
# 30 AMI ES meetings, 10-minute clips + rebased reference STMs, into
# ./amicorpus/clips (override with --out-dir; --group and --duration are flags):
python -m coro.bench.utils.make_ami_clip_set --ami-root ./amicorpus

coro-bench quality --clips-dir ./amicorpus/clips \
  --server-url http://127.0.0.1:8123 --out-dir run
```

Re-running skips meetings that are already materialized and never re-downloads,
so both runs score provably identical audio.

To reproduce a *row* of the 61-clip table, start a server with the backend you
want and everything else pinned, then attach the quality workload to it. Only
the first three variables change between rows:

```bash
CORO_BACKEND_ASR=faster-whisper CORO_MODEL_ASR=openai/whisper-large-v3-turbo \
CORO_ASR_COMPUTE_TYPE=float16 \
CORO_ASR_DEVICE=cuda CORO_BACKEND_DIARIZATION=nemo CORO_PIPELINE=full-memory \
  coro --port 8123
```

`CORO_MIN_TURN_WORDS` / `CORO_MIN_TURN_SECONDS` are deliberately left unset so
every row runs the default 2 word / 0.4 s **Minimum Turn Threshold**. The
parakeet row substitutes `CORO_BACKEND_ASR=onnx-asr`,
`CORO_MODEL_ASR=nemo-parakeet-tdt-0.6b-v3` and drops `CORO_ASR_COMPUTE_TYPE`
(fp32). Wait for `/health` to report **both** `ready` and `warmup_ready` before
starting the bench — `coro-bench quality` attaches over HTTP and does not
manage the server for you.

**ES**, not the built-in `sample` preset (IB4001 + IN1001, n=2, both
non-scenario). Every ES meeting has exactly four participants, matching the
default diarization model's hard four-speaker cap; the non-scenario groups can
exceed it, forcing the diarizer to merge two real speakers and inflating
speaker-attribution error for a reason no segmentation change can recover.

Other dataset loaders: `make_rttm_clip` (VoxConverse / diarization-only DER),
`make_common_voice_clips` (Common Voice WER). A trustworthy **Spanish** DER+WER
target (Albayzín-RTVE2020) is gated behind an RTVE licence — see the README
*Benchmark datasets* note; apply for access if you need Spanish numbers.
