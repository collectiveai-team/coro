r"""Extract parakeet-rnnt-1.1b-multilingual-prompt's ``prompt_kernel`` weights.

See ``README.md`` in this package for the full write-up (why this exists,
the license constraint on its output, and the checkpoint-loading cost it
avoids paying twice). This docstring stays short on purpose -- it is what
``--help`` shows.

Usage:
    uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_kernel_cache \\
        --checkpoint /path/to/parakeet-rnnt-1.1b-multilingual-prompt.nemo
    uv run --extra recipes --extra cpu -m coro.recipes.parakeet_prompt_kernel_cache \\
        --checkpoint /path/to/checkpoint.nemo --out-dir recipe-artifacts/parakeet_prompt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from coro.recipes.paths import RECIPE_ARTIFACTS_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a local parakeet-rnnt-1.1b-multilingual-prompt.nemo checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RECIPE_ARTIFACTS_DIR / "parakeet_prompt",
        help="Directory to write prompt_kernel_cache.npz/.json into.",
    )
    args = parser.parse_args(argv)

    import nemo.collections.asr as nemo_asr
    import torch

    print(f"loading checkpoint {args.checkpoint} ...")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(
        str(args.checkpoint), map_location=torch.device("cpu")
    ).eval()
    print(f"loaded in {time.time() - t0:.1f}s")

    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size
    blank_id = model.joint.num_classes_with_blank - 1
    prompt_dictionary = dict(model.cfg.model_defaults.get("prompt_dictionary") or {})
    weights = {
        name: param.detach().numpy() for name, param in model.prompt_kernel.state_dict().items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = args.out_dir / "prompt_kernel_cache.npz"
    cache_json = args.out_dir / "prompt_kernel_cache.json"
    np.savez(cache_npz, **weights)
    cache_json.write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "prompt_dictionary": prompt_dictionary,
            },
            indent=2,
        )
    )
    print(f"cached weights -> {cache_npz}")
    print(f"cached meta -> {cache_json}")
    print("prompt_dictionary:", prompt_dictionary)
    print("blank_id:", blank_id, "vocab_size:", vocab_size)
