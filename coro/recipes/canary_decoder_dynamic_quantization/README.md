# canary_decoder_dynamic_quantization

Dynamic INT8 quantization of Canary's `decoder_step.onnx`
([canary_split_decoder](../canary_split_decoder/README.md)'s output) — the
accepted `decoder_quantization` selector `onnx-canary-split` can load.

## What it does

Runs `onnxruntime.quantization.quantize_dynamic` (`op_types_to_quantize=
["MatMul"]`, no calibration data) twice, producing both weight-type
variants:

- `decoder_step.dynamic_v1_quint8.onnx` — **the accepted selector**.
- `decoder_step.dynamic_v1_qint8.onnx` — produced for comparison, **not**
  the accepted selector (see below).

## Worth knowing

- **Static QDQ was tried first on this exact graph and rejected.** The
  technique that works for this repo's RNNT/Canary *encoders* (calibrated,
  offline-scale static QDQ) does not work for an autoregressive decoder:
  real word-level WER damage was measured on all 3 held-out clips (e.g.
  "fell-Americans" for "fellow Americans"). Full write-up:
  `.journals/2026-09-04/2026-09-04_canary-decode-loop-rtf_decoder-quant-static-qdq-rejected-plus-research/`.

- **Why dynamic quantization instead, and why it isn't just a guess.**
  Three independent sources converged on it: CTranslate2 (the engine
  behind faster-whisper) quantizes Whisper's decoder weight-only/
  dynamically, never fixing activation scales from an offline calibration
  sample; `k2-fsa/sherpa-onnx`'s own Canary export scripts use plain
  `quantize_dynamic` for this exact model family; ONNX Runtime's own docs
  recommend dynamic quantization for transformer-based models and static
  quantization for CNNs. No calibration data is needed at all — that's the
  whole point of the technique, and why this recipe has no
  `--calibration-*` argument unlike
  [`parakeet_prompt_encoder_static_qdq`](../parakeet_prompt_encoder_static_qdq/README.md).

- **Measured result, not assumed.** Full 48-window mTEDx validation: norm
  cpWER 0.0529 vs the fp32 decoder's 0.0513 (+3.1% relative — an order of
  magnitude smaller than the rejected static-QDQ *encoder*'s +34.5%
  relative cost), RTFx 2.33 vs the fp32 split-decode baseline's 1.62
  (+43.8%). Full write-up:
  `.journals/2026-09-04/2026-09-04_canary-decode-loop-rtf_decoder-dynamic-quantization-accepted/`.

- **QUInt8, not QInt8, despite QInt8 looking better on a short-clip
  microbenchmark.** This recipe produces both variants for exactly this
  reason: `dynamic_v1_qint8` scored a *better* speedup on an isolated
  short-clip decoder-only microbenchmark (up to 6.15x vs QUInt8's 4.85x),
  but at full 48-window/144-step-deep production scale it actually came in
  **slower than fp32** (RTFx 1.55 vs the fp32 baseline's 1.62) with
  marginally worse quality too. Do not trust a short-clip quantization
  microbenchmark alone for this graph — always confirm against the full
  production-length windowing path before picking a selector.

## Usage

```sh
uv run --extra recipes --extra cpu -m coro.recipes.canary_decoder_dynamic_quantization
uv run --extra recipes --extra cpu -m coro.recipes.canary_decoder_dynamic_quantization \
    --source recipe-artifacts/canary_split/decoder_step.onnx \
    --out-dir recipe-artifacts/canary_decoder_int8
```

## Consumers

`coro/backends/asr/onnx_canary_split.py`'s `decoder_quantization="dynamic_v1_quint8"`
selector (also `coro/settings.py`'s `asr_decoder_quantization` Server
Setting) loads `decoder_step.dynamic_v1_quint8.onnx`. `None` loads the fp32
`decoder_step.onnx` from
[canary_split_decoder](../canary_split_decoder/README.md) instead, but the
default `canary-1b-v2` model slug fills `asr_decoder_quantization` with
`dynamic_v1_quint8` — this is one of the two quantizations the *default* ASR
Backend Provider ships with (see ADR 0019); explicit `fp32` opts back out.

## Not migrated here (stay in `.tmp/`, rejected)

- `.tmp/quantize_canary_decoder.py` — the rejected static-QDQ decoder
  attempt referenced above.
- `.tmp/quantize_canary_decoder_weightonly.py` — weight-only INT8 (passed
  the 3-clip screen but with a lower speedup ceiling than dynamic
  quantization; not carried forward to full-scale validation).
