# canary_encoder_static_qdq

Static QDQ INT8 quantization of the **Canary encoder only**
(`encoder-model.onnx` from `istupakov/canary-1b-v2-onnx`), producing the
`encoder-model.static_qdq_v4_pct_excl.onnx` artifact
[`onnx-canary-split`](../../backends/asr/onnx_canary_split.py)'s
`quantization` selector loads.

The decoder side is a separate concern with a separate answer — see
[canary_decoder_dynamic_quantization](../canary_decoder_dynamic_quantization/README.md).
The two compose; they are not alternatives.

## What it does

1. `quant_pre_process` (shape inference) on the fp32 encoder, cached.
2. **Percentile** activation calibration at 99.999% over
   [calibration_corpus](../calibration_corpus/README.md)'s 40 clips
   (8 each of es/en/fr/de/pt, 4.3–14.4 s), collected one clip at a time.
   The computed ranges are cached to JSON so a rebuild that changes only the
   exclusion list skips the expensive pass.
3. `quantize_static` restricted to `Conv`/`MatMul`/`Gemm`, `per_channel=True`,
   `reduce_range=True`, `QUInt8` activations / `QInt8` weights, with 32
   measured-worst nodes left in fp32.

## Why this technique — and what the first attempt got wrong

An earlier attempt (`.tmp/quantize_canary_encoder.py`, selector
`static_qdq_v3`) used the same op-type restriction and the same corpus, but
**MinMax activation calibration** and **no per-node exclusions**. It passed a
3-clip screen with "zero word-level quality loss" and was then rejected at
full 48-window mTEDx scale: norm cpWER rose from the fp32 encoder's 0.0513 to
**0.0690, +34.5% relative**. Two things were wrong with it, and
`onnxruntime.quantization.qdq_loss_debug` — flagged by two prior sessions and
never actually run until this one — measured both.

**1. MinMax stretches every scale to the worst outlier.** The measurement
(`.tmp/canary_encoder_qdq_sensitivity.py`, comparing fp32 and QDQ activations
over held-out mTEDx audio) found the v3 graph's per-tensor quantization SNR
had a **median of 32.7 dB** across 973 quantized activations, with 53 tensors
below 20 dB. A well-conditioned INT8 tensor lands above 40 dB; each doubling
of an over-wide range costs 6 dB. Switching to Percentile 99.999 moved the
median to 34.0 dB, the 10th percentile from 22.8 dB to 25.4 dB, and the
*accumulated* cross-model SNR median from 10.6 dB to 13.7 dB.

**2. The residual damage is concentrated, so all-or-nothing was the wrong
question.** Even after percentile calibration, a small set of nodes stayed
catastrophically noisy — 6–22 dB against a 34 dB median — and they are not
scattered: they are the **convolution module of the last eight conformer
layers** (`conv1d_74`…`conv1d_95`, the layer-24..31 pointwise projections,
plus their depthwise partners `node_Conv_39xx`/`node_Conv_40xx`) and the
late-layer attention/feed-forward MatMuls. Excluding those 32 nodes is ~7% of
the quantized nodes and leaves the rest INT8.

Calibration *corpus* composition was investigated and ruled out as the
suspect: v3's corpus was already 40 clips across five languages at 4.3–14.4 s,
and the encoder never sees `target_lang` (see
[calibration_corpus](../calibration_corpus/README.md)), so cross-language
diversity was never the missing ingredient. The corpus is unchanged here; only
the calibration *method* and the exclusion list differ.

## Measured result (full 48-window mTEDx, `mtedx-HLIJkmy3vy8`)

Same corpus, windowing path and cpWER scoring as every other gate in this
investigation. `fp32` is an in-run control, not a quoted historical number —
it reproduces the recorded baseline exactly, which validates the harness.

| encoder | norm cpWER | vs fp32 | foreign_sub_rate | RTFx |
|---|---:|---:|---:|---:|
| fp32 (control) | 0.0513 | — | 0.0 | 1.2112 |
| `static_qdq_v3` — MinMax, no exclusions (rejected) | 0.0690 | +34.5% | — | — |
| `static_qdq_v4_pct` — Percentile, no exclusions | 0.0615 | +19.9% | 0.0137 | 1.2877 |
| `static_qdq_v4_pct_excl` — **both** | **0.0508** | **−1.0%** | **0.0** | **1.2600** |

`function_word_hits` was `{}` on every arm — no arm introduced
foreign-language drift.

**Both changes were necessary and neither was sufficient.** Better calibration
alone cuts the damage roughly in half (+34.5% → +19.9%) but still fails the
bar; it is the per-node exclusions, chosen from measured SNR rather than
guessed, that close the remaining gap and land slightly *ahead* of fp32.
That ordering is worth remembering: the calibration fix is the obvious one to
reach for, and on its own it would have produced a second rejection.

The speed gain is real but modest: **+4.0% RTFx at −1.0% cpWER**. The
exclusions that buy back the quality are exactly the heavy late-layer
convolutions, so this recovers less throughput than the unexcluded variant's
+6.3%. Ship it because it is strictly better than fp32 on both axes, not
because it is fast.

The absolute RTFx figures here are lower than the 1.6212 recorded for the same
fp32 arm in an earlier session — that run was on an otherwise-idle host. Only
compare arms *within* a run.

## Worth knowing

- **`op_types_to_quantize` is not a performance tuning knob.** Unrestricted
  `quantize_static` also wraps the `encoder_mask` arithmetic subgraph
  (`Cast`/`Floor`/`Range`/`Sub`/`Less`, which derives output length from input
  length) in QDQ nodes, corrupting exact integer/length math with lossy int8
  rounding. This was diagnosed on the RNNT encoder first; both encoder recipes
  carry the same restriction from the start.
- **Histogram calibration must be chunked at exactly one clip.** ORT's
  `HistogramCollector` accumulates every clip's intermediate tensors before
  merging, which exhausts memory on a 1 B-parameter encoder. ORT's
  `CalibStridedMinMax` extra option (a generic chunk size despite the name)
  drives collection incrementally — but at a chunk size above 1 the collector
  calls `np.asarray()` over tensors from clips of *different durations* and
  raises on the inhomogeneous shape. Stride 1 is the only working setting for
  this corpus.
- **The exclusion list is pinned, not re-derived at build time.** Rebuilding
  it requires an already-quantized graph to measure against, plus probe audio;
  baking that into the recipe would make the accepted artifact
  non-deterministic. `.tmp/canary_encoder_qdq_sensitivity.py` is the research
  script that produced it — regenerate the list with that script if the
  upstream fp32 encoder ever changes.
- **`reduce_range=True` is required on this host class** (AVX2, no VNNI),
  per ORT's own guidance for the U8S8 path.

## Usage

```sh
uv run --extra recipes --extra cpu -m coro.recipes.canary_encoder_static_qdq \
    --source /path/to/encoder-model.onnx
uv run --extra recipes --extra cpu -m coro.recipes.canary_encoder_static_qdq \
    --source /path/to/encoder-model.onnx \
    --calibration-manifest recipe-artifacts/calibration_corpus/calibration_manifest.json \
    --out-dir recipe-artifacts/canary_split
```

Requires [calibration_corpus](../calibration_corpus/README.md) to have been
built first.

## Consumers

[`onnx-canary-split`](../../backends/asr/onnx_canary_split.py), via
`Settings.asr_quantization="static_qdq_v4_pct_excl"`.

## License

`nvidia/canary-1b-v2` is **CC-BY-4.0** — unlike the Parakeet-Prompt recipes in
this package, neither this recipe's source checkpoint nor its output carries a
redistribution or NIM/AI-Enterprise production-use restriction.
