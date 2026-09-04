# canary_split_decoder

Splits the fused Canary decoder ONNX graph at the cross-attention K/V
frontier, producing the two graphs behind the shipped `onnx-canary-split`
backend (issue #64's Canary decode-loop RTF fix, commit `20a5f4a`).

## What it does

`decoder-model.onnx` (from `istupakov/canary-1b-v2-onnx`) has no
cross-attention key/value cache: exactly 16 nodes consume the
`encoder_embeddings` graph input directly (8 decoder layers x
`{key_net, value_net}` cross-attention projections, each a `[1024, 1024]`
MatMul under `/_decoder/layers.{0..7}/second_sub_layer/{key,value}_net/`),
and the fused graph recomputes all 16 from scratch on every decode step even
though their output is identical for every step within a window.

The recipe:

1. Downloads `decoder-model.onnx` (cached via `huggingface_hub`).
2. Runs ONNX shape inference on it.
3. Uses `onnx.utils.extract_model` to cut it into two graphs:

   ```text
   xattn_kv.onnx      in=[encoder_embeddings]                          -> 16 K/V tensors
                       (called once per window)
   decoder_step.onnx  in=[input_ids, encoder_mask, decoder_mems,
                           16 K/V tensors]  -> [logits, decoder_hidden_states]
                       (called once per decode step)
   ```

4. Verifies the split composes back to the fused graph, on **real** encoder
   output (not synthetic random input), over >=3 sequential decode steps
   with growing `decoder_mems` — checking only step 0 would miss a bug
   where cached K/V silently goes stale as the self-attention cache grows.

## Worth knowing

- **The cut point is not the obvious one.** Graph inspection (against the
  real ONNX graph, not just cited from a prior session's notes) found the
  maximal hoistable cut is *not* the 16 MatMul outputs directly — it's the
  16 `/_decoder/layers.{0..7}/second_sub_layer/Transpose_{2,3}_output_0`
  tensors one level downstream:

  ```text
  MatMul[1024,1024] -> Add(bias) -> Reshape -> Transpose  =>  Transpose_{2,3}_output_0
  ```

  Per layer, `Transpose_3_output_0` (key, scaled by the attention `Div`)
  feeds `MatMul` (attention scores, against the per-step query);
  `Transpose_2_output_0` (value) feeds `MatMul_1` (against the softmax'd
  scores). Both are consumed only by step-dependent computation and both
  are pure functions of `encoder_embeddings` alone — verified via backward
  ancestor traversal from each of the 16 tensors: the only graph input
  reached is `encoder_embeddings` (`encoder_mask`'s single direct consumer,
  `/_decoder/Cast_2`, is unrelated to the K/V projections).

- **The known symbolic-shape risk did not occur, but here's the fallback
  if it does on a future re-export.** The ticket's spec flagged a risk:
  `infer_shapes` returning symbolic `unk__` dimension names for the
  frontier tensors, which would make `extract_model`'s validation reject
  the cut. This did **not** happen for this graph (`decoder-model.onnx`
  has no external-data sidecar, unlike the encoder; a 649M-parameter,
  nodes-only IR, so plain `infer_shapes` + `extract_model` just worked). If
  a future re-export of this checkpoint does hit that failure mode,
  construct explicit `value_info` for the 16 frontier tensors by hand (the
  op is `Transpose` of a `Reshape` of an `Add` of a `MatMul[1024,1024]`
  against `encoder_embeddings [batch, encoded_len, 1024]`) rather than
  falling back to the shallower MatMul-output cut.

- **Verified end-to-end reproducibility, not just "should work."** Re-ran
  this recipe standalone after migrating it out of `.tmp/`: `max|Δ| =
  0.000e+00` composition against the fused graph, byte-for-byte identical
  to the original research script's result.

- **Logic is unchanged from the original `.tmp/split_canary_decoder.py`**
  research script — only paths were parameterized (`--out-dir`,
  `--holdout-wav`, `--verify-steps`) when this was promoted out of `.tmp/`.

## Usage

```sh
uv run --extra recipes --extra cpu -m coro.recipes.canary_split_decoder
uv run --extra recipes --extra cpu -m coro.recipes.canary_split_decoder \
    --out-dir recipe-artifacts/canary_split --verify-steps 4
```

## Consumers

`coro/backends/asr/onnx_canary_split.py` loads this recipe's
`xattn_kv.onnx` and `decoder_step.onnx` unconditionally (they are not
optional/quantized artifacts — see
[`coro.recipes.canary_decoder_dynamic_quantization`](../canary_decoder_dynamic_quantization/README.md)
for the optional quantized `decoder_step.<selector>.onnx` variant this
backend can load instead).

## Further reading

- `.scratch/canary-decode-loop-rtf/PRD.md` (issue #64) and
  `.scratch/issue-64-language-constrained-asr/findings.md` — the full
  investigation this recipe's output implements.
