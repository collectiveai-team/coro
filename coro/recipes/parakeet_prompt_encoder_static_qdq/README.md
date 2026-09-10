# parakeet_prompt_encoder_static_qdq

Static QDQ INT8 quantization of the Parakeet-Prompt encoder — the
current-best encoder artifact `onnx-parakeet-prompt` loads
(`encoder-encoder.static_qdq_v3.onnx`).

## What it does

1. Pre-processes the fp32 `encoder-encoder.onnx` (shape inference via
   `quant_pre_process`), caching the result so re-runs against an
   unchanged source skip this step.
2. Loads a calibration corpus (see
   [calibration_corpus](../calibration_corpus/README.md)) and feeds real
   mel-spectrogram features through the encoder's `audio_signal`/`length`
   input contract for MinMax calibration.
3. Runs `onnxruntime.quantization.quantize_static` with non-VNNI CPU
   parameters (`activation_type=QUInt8, weight_type=QInt8,
   reduce_range=True, per_channel=True`), restricted to
   `op_types_to_quantize=["Conv", "MatMul", "Gemm"]`.

`decoder_joint` is deliberately left untouched (fp32) — research found
it's ~1-2% of total time for this model family, not worth the
quantization risk/effort.

## Worth knowing

- **This host is AVX2-only, no VNNI** (confirmed via `lscpu`) — the
  `activation_type=QUInt8, weight_type=QInt8, reduce_range=True`
  combination is ONNX Runtime's own documented recommendation for that
  case, not an arbitrary choice.

- **`op_types_to_quantize` and `per_channel=True` are not defaults — they
  fix a real, diagnosed bug, and removing them re-introduces it.** An
  earlier *unrestricted* first attempt (no `op_types_to_quantize`,
  `per_channel=False`) let `quantize_static` insert
  QuantizeLinear/DequantizeLinear into the encoder's `encoded_lengths`
  arithmetic subgraph — Cast/Add/Sub/Div nodes that derive output length
  from input length via conv-stride math. Confirmed via graph-topology
  tracing: 0 quant nodes in that subgraph for fp32, 22
  QuantizeLinear+22 DequantizeLinear pairs for the unrestricted
  static-QDQ export. Turning exact length arithmetic into lossy int8
  rounding produced a **wrong** `encoded_lengths` (71 vs fp32's 74) and,
  combined with per-tensor (not per-channel) MinMax scale on a
  1024-dim/42-layer encoder and a 3-clip calibration set, an
  86%-relative-magnitude corruption in the *valid* frames too — which
  caused a forced-language decode to collapse to **zero non-blank
  emissions**. This was diagnosed via the `diagnose` skill
  (DEBUG-e64c) and was *not* a "cross-language calibration gap" as first
  suspected — the encoder never sees the target language at all, only
  audio. The fix applied here from the start: restrict quantization to
  the actual compute-heavy ops
  (`op_types_to_quantize=["Conv", "MatMul", "Gemm"]`), leaving
  arithmetic/control nodes in full precision, paired with
  `per_channel=True` (per ORT's own non-VNNI guidance).

- **Needs a calibration corpus first.** Run
  [calibration_corpus](../calibration_corpus/README.md) before this recipe
  if `--calibration-manifest`'s default location doesn't already exist.

## Usage

```sh
uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_encoder_static_qdq \
    --source /path/to/encoder-encoder.onnx
uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_encoder_static_qdq \
    --source /path/to/encoder-encoder.onnx \
    --calibration-manifest recipe-artifacts/calibration_corpus/calibration_manifest.json \
    --out-dir recipe-artifacts/parakeet_prompt
```

`--source` is not produced by any recipe in this package — it's `onnx_asr`'s
own encoder export for this checkpoint, acquired separately.

## Consumers

`coro/backends/asr/onnx_parakeet_prompt.py`'s
`quantization="static_qdq_v3"` selector loads
`encoder-encoder.static_qdq_v3.onnx`.
