r"""Split the fused Canary decoder ONNX graph at the cross-attention K/V frontier.

See ``README.md`` in this package for the full write-up (why this cut,
what it produces, and what to watch for on a future re-export). This
docstring stays short on purpose -- it is what ``--help`` shows.

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.canary_split_decoder
    uv run --extra recipes --extra cpu -m coro.recipes.canary_split_decoder \\
        --out-dir recipe-artifacts/canary_split --verify-steps 4
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from coro.cli_support import invoke, new_app, out_console
from coro.core.logger import get_logger
from coro.recipes.paths import RECIPE_ARTIFACTS_DIR

log = get_logger(__name__)

N_LAYERS = 8

# Cut points: per layer, key (Transpose_3) then value (Transpose_2) -- see
# README.md for why these are the maximal hoistable frontier.
KV_TENSORS: list[str] = []
for _layer in range(N_LAYERS):
    KV_TENSORS.append(f"/_decoder/layers.{_layer}/second_sub_layer/Transpose_3_output_0")
    KV_TENSORS.append(f"/_decoder/layers.{_layer}/second_sub_layer/Transpose_2_output_0")


def resolve_decoder_path() -> Path:
    """Locate the cached `decoder-model.onnx`, downloading only if missing."""
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            "istupakov/canary-1b-v2-onnx",
            allow_patterns=["decoder-model.onnx", "config.json", "vocab.txt"],
        )
    )
    return snapshot_dir / "decoder-model.onnx"


@dataclass(frozen=True)
class SplitPaths:
    """Paths to the two graphs extracted from the fused decoder."""

    xattn_kv: Path
    decoder_step: Path


def build(src: Path, out_dir: Path) -> SplitPaths:
    """Extract xattn_kv.onnx and decoder_step.onnx from the fused decoder graph."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("loading", path=str(src))
    model = onnx.load(str(src))
    log.info("loaded", nodes=len(model.graph.node), initializers=len(model.graph.initializer))

    log.info("running shape inference")
    inferred = onnx.shape_inference.infer_shapes(model)
    inferred_path = out_dir / "decoder.inferred.onnx"
    onnx.save(inferred, str(inferred_path))

    specs = {
        "xattn_kv": (["encoder_embeddings"], KV_TENSORS),
        "decoder_step": (
            ["input_ids", "encoder_mask", "decoder_mems", *KV_TENSORS],
            ["logits", "decoder_hidden_states"],
        ),
    }

    dsts: dict[str, Path] = {}
    for name, (ins, outs) in specs.items():
        dst = out_dir / f"{name}.onnx"
        log.info(
            "extracting",
            graph=f"{name}.onnx",
            inputs=ins if len(ins) < 5 else [*ins[:3], "..."],
            outputs=outs if len(outs) < 5 else [*outs[:3], "..."],
        )
        onnx.utils.extract_model(str(inferred_path), str(dst), ins, outs)
        g = onnx.load(str(dst), load_external_data=False).graph
        n_params = sum(int(np.prod(list(i.dims))) for i in g.initializer)
        log.info("extracted", graph=name, nodes=len(g.node), params_m=round(n_params / 1e6, 2))
        dsts[name] = dst

    inferred_path.unlink(missing_ok=True)
    return SplitPaths(xattn_kv=dsts["xattn_kv"], decoder_step=dsts["decoder_step"])


def _real_encoder_embeddings(holdout_wav: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run the actual Canary encoder on a real clip -- not synthetic random input."""
    import soundfile as sf

    import onnx_asr

    audio, sr = sf.read(holdout_wav, dtype="float32")
    if sr != 16000:
        msg = f"{holdout_wav} is {sr} Hz, expected 16000"
        raise ValueError(msg)

    model = onnx_asr.load_model("nemo-canary-1b-v2", providers=["CPUExecutionProvider"])
    asr = model.asr
    # Deliberately reaching into onnx_asr's private preprocessing/encode
    # methods -- same access pattern the backend adapters use on their own
    # subclasses, but this recipe drives a plain (non-subclassed) instance.
    features, features_lens = asr._preprocessor(  # pyrefly: ignore[missing-attribute]
        audio[None, :].astype(np.float32), np.array([audio.shape[0]], dtype=np.int64)
    )
    encoder_embeddings, encoder_mask = asr._encode(  # pyrefly: ignore[missing-attribute]
        features, features_lens
    )
    return np.asarray(encoder_embeddings), np.asarray(encoder_mask)


def verify(
    fused_path: Path, split_paths: SplitPaths, holdout_wav: Path, *, n_steps: int = 4
) -> bool:
    """Run fused vs split composition over n_steps sequential decode steps.

    decoder_mems grows each step (self-attention cache), exactly like the
    real ``_decoding`` loop -- checking only step 0 would miss a bug where
    cached K/V silently goes stale as the self-attention cache grows.
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    fused = ort.InferenceSession(str(fused_path), so, providers=["CPUExecutionProvider"])
    xattn_kv = ort.InferenceSession(
        str(split_paths.xattn_kv), so, providers=["CPUExecutionProvider"]
    )
    decoder_step = ort.InferenceSession(
        str(split_paths.decoder_step), so, providers=["CPUExecutionProvider"]
    )

    log.info("running real Canary encoder for real encoder_embeddings/encoder_mask")
    encoder_embeddings, encoder_mask = _real_encoder_embeddings(holdout_wav)
    log.info(
        "encoder output",
        encoder_embeddings_shape=encoder_embeddings.shape,
        encoder_mask_shape=encoder_mask.shape,
    )

    log.info("running xattn_kv.onnx once (per-window cost)")
    t0 = time.perf_counter()
    kv_outputs = xattn_kv.run(KV_TENSORS, {"encoder_embeddings": encoder_embeddings})
    xattn_kv_ms = (time.perf_counter() - t0) * 1e3
    kv_feed = dict(zip(KV_TENSORS, kv_outputs, strict=True))
    log.info("xattn_kv.onnx ran", ms=round(xattn_kv_ms, 2))

    batch_size = encoder_embeddings.shape[0]
    input_ids = np.array([[1, 2, 3]], dtype=np.int64)  # arbitrary prefix, grown each step
    decoder_mems = np.empty((10, batch_size, 0, 1024), dtype=np.float32)

    ok = True
    for step in range(n_steps):
        fused_out = fused.run(
            ["logits", "decoder_hidden_states"],
            {
                "input_ids": input_ids if decoder_mems.shape[2] == 0 else input_ids[:, -1:],
                "encoder_embeddings": encoder_embeddings,
                "encoder_mask": encoder_mask,
                "decoder_mems": decoder_mems,
            },
        )
        fused_logits, fused_mems = np.asarray(fused_out[0]), np.asarray(fused_out[1])
        split_out = decoder_step.run(
            ["logits", "decoder_hidden_states"],
            {
                "input_ids": input_ids if decoder_mems.shape[2] == 0 else input_ids[:, -1:],
                "encoder_mask": encoder_mask,
                "decoder_mems": decoder_mems,
                **kv_feed,
            },
        )
        split_logits, split_mems = np.asarray(split_out[0]), np.asarray(split_out[1])
        d_logits = float(np.max(np.abs(fused_logits - split_logits)))
        d_mems = float(np.max(np.abs(fused_mems - split_mems)))
        step_ok = d_logits == 0.0 and d_mems == 0.0
        ok = ok and step_ok
        log.info(
            "verified step",
            step=step,
            decoder_mems_shape=decoder_mems.shape,
            logits_max_abs_diff=d_logits,
            decoder_hidden_states_max_abs_diff=d_mems,
            ok=step_ok,
        )

        # advance exactly like the real _decoding loop: append next token, mems grows.
        next_token = int(np.argmax(fused_logits[:, -1], axis=-1)[0])
        input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=-1)
        decoder_mems = fused_mems

    return ok


app = new_app("canary_split_decoder", help=__doc__.splitlines()[0])


@app.default
def _split(
    *,
    out_dir: Path = RECIPE_ARTIFACTS_DIR / "canary_split",
    holdout_wav: Path = Path("coro/bench/data/jfk.wav"),
    verify_steps: int = 4,
) -> None:
    """Split the fused decoder and verify the split against it.

    Args:
        out_dir: Directory to write xattn_kv.onnx/decoder_step.onnx into.
        holdout_wav: Real clip used to derive encoder_embeddings/encoder_mask for verification.
        verify_steps: Sequential decode steps to verify.

    Raises:
        SystemExit: Code 0 when the split matches the fused graph, 1 otherwise.

    """
    fused_path = resolve_decoder_path()
    log.info("splitting", decoder=fused_path.name)
    paths = build(fused_path, out_dir)

    log.info("verifying composition against the fused graph (real encoder output, >=3 steps)")
    ok = verify(fused_path, paths, holdout_wav, n_steps=verify_steps)
    out_console.out(f"\nSPLIT VALID: {ok}")
    raise SystemExit(0 if ok else 1)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Canary decoder split recipe."""
    sys.stdout.reconfigure(line_buffering=True)  # pyrefly: ignore[missing-attribute]
    invoke(app, argv)
