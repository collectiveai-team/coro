# Canary-1b-v2 (INT8/INT8) Is the Default ASR Model Selection

The previous default, `onnx-asr` / `nemo-parakeet-tdt-0.6b-v3`, does implicit, per-frame language identification with no way to constrain it. On single-language recordings it routinely switches language mid-file: the 48-window mTEDx Spanish gate measured **45 English function-word intrusions across 8 of 48 windows** (one window alone carried 14) against **zero** with forced-language Canary (`.scratch/issue-64-language-constrained-asr/findings.md`). For a Spanish-judicial-hearing product that is disqualifying regardless of Parakeet's throughput advantage — a transcript that silently drifts into the wrong language is worse than one that is merely slower — and the package stays language-agnostic, so the fix could not be a product-specific patch. Canary-1b-v2 (INT8 encoder + INT8 decoder, `onnx-canary-split`) is quality-validated, CC-BY-4.0, and now fixes that: **`onnx-canary-split` becomes the default ASR Backend Provider**, fetched by default from `collectiveai/canary-1b-v2-onnx-split-int8` (the `canary-1b-v2` Model Slug, ADR 0020).

This is a **deliberate, accepted throughput trade**, not a claim that Canary is faster. Parakeet's per-frame LID is the disqualifier; its speed was never in question and stays available via `--model-asr parakeet-tdt-0.6b-v3`. See the "Settled decisions" note in `.scratch/canary-default/PRD.md` and the Non-goals below.

## What made Canary shippable as a default

Three separate blockers had to close before Canary could carry production traffic at all, each already validated on the standing 48-window mTEDx gate (`.tmp/mtedx-drift`, 22.4 min) before this ADR:

- **Speed.** The fused decoder recomputed 16 cross-attention K/V tensors on every decode step despite them being constant within a window (`.scratch/canary-decode-loop-rtf/PRD.md`, issue #64). A graph-surgery split (`xattn_kv.onnx` + `decoder_step.onnx`, `max|Δ| = 0.000e+00` against the fused graph) removed that, then two independent INT8 quantizations closed the rest of the gap: encoder `static_qdq_v4_pct_excl` (percentile calibration + a measured node-exclusion list — naive MinMax calibration was tried and rejected at +34.5% relative cpWER first) and decoder `dynamic_v1_quint8` (dynamic-range INT8; static-QDQ was tried on the decoder too and rejected for real word-level WER damage on the autoregressive loop). See `coro/backends/asr/onnx_canary_split.py`'s module docstring for the full numbers and the two rejected alternatives.
- **Concurrency.** The split-decode adapter kept its K/V cache as plain instance state, serialising the backend to one inference at a time. Moving it to a `threading.local()` holder (each `transcribe_pcm` call already runs on its own worker thread via `asyncio.to_thread`) made overlapping windows race-free without locking, and the backend picked up the same auto-sized `AdmissionController` `onnx-asr` uses.
- **Artifact distribution.** The backend required a hand-built local directory. It now resolves `model_asr` as a Hugging Face repo id too (`collectiveai/canary-1b-v2-onnx-split-int8`, public, CC-BY-4.0), filtered to exactly the quantizations selected — the default INT8/INT8 pull is ≈1.29 GB, not the 609 MB fp32 decoder it never needs, and the fp32 encoder selector resolves from a second repo (`istupakov/canary-1b-v2-onnx`) since the default repo does not hold it.

None of that closed the language-control gap by itself — Parakeet's *speed* was never the blocker — but shipping a *default* on an admission-serialised backend with no reproducible artifact path was not viable regardless of quality, so both had to land before the default could move.

## The language strategy: forced > sticky auto-LID > fallback

Three layers, in strict precedence order, on every window:

1. **Forced.** A request `language` always wins outright and is normalised (`es-US` → `es`; underscores, case, whitespace) against the `<|xx|>` tokens actually present in the loaded vocab — never a hardcoded list. An unsupported language (e.g. `ja`, which this checkpoint does not carry) is a request-surface HTTP 400 (`param="language"`, OpenAI-style body) naming the supported set, on all three negotiating surfaces (OpenAI multipart + SSE, Deepgram batch, Deepgram live websocket at negotiate time) — never a `KeyError`, never a silent 500.
2. **Sticky auto-LID.** With no request language, the adapter's own checkpoint predicts the source language from a 3-token partial prompt (NeMo's Canary2 `user_partial` role, `<|startofcontext|><|startoftranscript|>`): two greedy decoder steps, the second of which is the source-language slot. Ticket 01's probe measured **100% accuracy across 64 speech windows** (48/48 Spanish mTEDx windows, 4/4 each of English/French/German/Portuguese FLEURS clips) through both INT8 graphs, at a marginal cost of ~55 ms/window (reusing the transcription pass's own encoder output and K/V) — see `.scratch/canary-default/findings-lid-probe.md`. The **first window whose detection succeeds fixes the language for every later window of the same request/connection** (`LanguageState`, `coro/pipelines/windowing.py`); this sticky, request-scoped state lives with the pipeline call, not inside the shared, concurrent adapter instance. All three pipelines (Full-Memory, Streaming, Live) carry it. Non-speech windows were measured to **always** emit *some* language token (never a "no language" signal — digital silence deterministically maps to `<|en|>`, noise and real gaps to arbitrary codes), which is why stickiness keys on whichever window's detection succeeds first rather than trusting window 1 unconditionally when it might be leading silence.
3. **Fallback.** If no window in the whole request ever yields a language (not observed in probing, but handled), or before auto-LID resolves anything, decoding uses `asr_fallback_language` (`CORO_ASR_FALLBACK_LANGUAGE`, default `en`) — the same value Server Warmup passes explicitly. This replaces the previous behaviour of an unconditional, undocumented `<|en|>` baked into the decode prefix.

The `<|unklang|>` alternative (decode with an explicit "unknown language" source token) was probed and **decisively rejected**: on both the INT8 and fp32 checkpoint, it makes the decoder emit a space then immediate end-of-text — empty transcripts, WER 1.0, on every window tested. No deployment configuration of these artifacts can use it; it is a property of the exported checkpoint, not a quantization artifact.

Detected language is reported in `verbose_json.language` (and the Deepgram live socket's closing `Metadata` frame gains an optional `detected_language` field, since that surface previously had no language-reporting slot at all) instead of the previous `"unknown"`.

## The combined-configuration gate

Each accepted quantization was validated independently against fp32 on its own axis (encoder alone, decoder alone). The **combined** configuration — INT8 encoder + INT8 decoder, the actual shipped default — and the **auto-LID** arm (paying the sticky per-request detection cost once, not per window) had no single-process, interleaved measurement until this ADR's own gate. Three arms, interleaved window-by-window on the same 48-window mTEDx corpus so thermal drift hits all arms equally:

| arm | norm cpWER | RTFx |
|---|---:|---:|
| fp32/fp32, forced `es` (reference) | 0.0513 | 2.20× |
| INT8/INT8, forced `es` (default, explicit language) | 0.0526 | 4.45× |
| INT8/INT8, sticky auto-LID (default, no language) | 0.0526 | 4.36× |

Auto-LID detected `es` on all 48 of 48 windows with zero misses (sticky: only the first window's detection call actually runs; it fixed the language for the rest of the request), and the auto-LID and forced-`es` transcripts were **byte-identical**. INT8/INT8 combined **doubles RTFx** over fp32/fp32 (2.20× → 4.45×) while cpWER stays within run-to-run noise (0.0526 vs 0.0513) — both quantizations were independently accepted for being small-but-real wins, unlike `parakeet-tdt-0.6b-v3`'s `int8`, which trades WER for memory with no throughput gain. Auto-LID's one-time per-request detection cost is invisible at this scale (4.45× vs 4.36×): one extra encoder pass on window 1, amortised over 22.4 minutes of audio. Full methodology, wall-stats log and thresholds: `docs/benchmark.md#canary-default-int8int8-combined-quantization--auto-lid-gate`.

## The one-extra-load-per-run cost, accepted

Making an auto-LID-capable backend the default means `coro run`/a fresh `LazyASRAdapter` with no explicit `--language` now pays exactly one model load per invocation even on an otherwise fully-cached run, because `detect_language` must run before a window's cache key is even computable. This is measured and accepted, not glossed over: `tests/test_offline_run.py::test_an_auto_lid_capable_default_still_loads_once_per_run_for_detection` documents and asserts it explicitly (exactly one load, never more; the cache-hit report already proves the actual *transcription* is never redone), and its sibling `test_a_fully_cached_run_never_builds_the_adapter` now passes `--language es` to isolate the property it actually names (window-cache laziness) from this cost.

## Licensing

`nvidia/canary-1b-v2` is **CC-BY-4.0** — unlike `parakeet-rnnt-1.1b-multilingual-prompt` (NVIDIA Community Model License, NIM-gated, driven by the `nemo` and `onnx-parakeet-prompt` backends, both of which remain comparative-reference-only and never a default), Canary carries no redistribution or NIM/AI-Enterprise production-use restriction. This is a hard requirement, not a preference: a default backend that could not legally ship without an external vendor runtime dependency was never a candidate regardless of quality.

## Non-goals (unchanged from the PRD)

- Beating Parakeet's RTFx. It was never a speed contest.
- Removing `onnx-asr`, `nemo`, `onnx-parakeet-prompt`, `onnx-genai`, or `faster-whisper` — every existing backend keeps working with its current explicit flags.
- Per-window (Parakeet-style) language switching in any form — this is the behaviour being removed, not one being preserved as an option.
- Retuning ASR Windowing geometry (window/overlap seconds), VAD, or diarization behaviour.

## Superseded

`.scratch/canary-decode-loop-rtf/PRD.md`'s non-goal "not a change to the default (`onnx-asr` stays the default)" is superseded by this ADR — see the pointer added to that PRD. Its RTF/quality measurements for the split-decode graph and encoder INT8 in isolation still stand; only that older non-goal does not.
