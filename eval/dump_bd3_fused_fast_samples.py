"""Generate self-describing TinyStories dumps with the fused BD3-LM sampler."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bcfm-textdump/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "baseline/BD3LM_best.pt")
    parser.add_argument(
        "--baseline-code-root", type=Path,
        default=ROOT / "external/bcfm_baselines_uploaded_20260807",
    )
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--include-first-hitting", action="store_true")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=("fp32_tf32", "bf16"), default="fp32_tf32",
    )
    parser.add_argument(
        "--compile-mode", choices=("reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--label", default="BD3LM_fused_fast")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    os.environ["BCFM_INFERENCE_COMPILE_MODE"] = args.compile_mode

    code_root = args.baseline_code_root.resolve()
    sys.path.insert(0, str(code_root))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.bd3lm_fast import generate_fast_bd3
    from bcfm_baselines.models.generative_text import GenerationConfig
    import transformers

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    print(
        f"[config] experiment=bd3lm_fused_fast_quality device={args.device} "
        f"precision={args.precision} compile_mode={args.compile_mode} "
        f"seed={args.seed} nfe={args.nfe} "
        f"n_samples={args.n_samples} batch_size={args.batch_size} "
        f"cache_mode=dynamic_kv+compiled_cuda_graph+fused_transition output={out}",
        flush=True,
    )
    print(f"[stage=model_load] checkpoint={checkpoint} sha256={_sha256(checkpoint)}", flush=True)
    model, payload = load_checkpoint_model(checkpoint, args.device)
    if payload.get("step") != 100000 or payload["config"]["dataset"]["name"] != "tinystories":
        raise RuntimeError("unexpected BD3-LM checkpoint provenance")
    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    print(
        f"[stage=model_ready] gpu={torch.cuda.get_device_name(0)} "
        f"parameter_dtype={next(model.parameters()).dtype} step={payload['step']} "
        f"block_size={model.block_size}", flush=True,
    )

    specs = [(f"sample_nfe{nfe}", nfe, False) for nfe in args.nfe]
    if args.include_first_hitting:
        specs.append(("sample_nfe256", 256, True))
    points: list[dict] = []
    token_arrays: dict[str, np.ndarray] = {}
    total_start = time.perf_counter()
    for point_index, (point_id, steps, first_hitting) in enumerate(specs):
        stage_start = time.perf_counter()
        batches: list[Tensor] = []
        produced = 0
        n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
        print(
            f"[stage=sampling] point={point_id} progress={point_index + 1}/{len(specs)} "
            f"first_hitting={first_hitting} batches={n_batches}", flush=True,
        )
        for batch_index in range(n_batches):
            size = min(args.batch_size, args.n_samples - produced)
            generation = GenerationConfig(
                length=256,
                batch_size=size,
                strategy="sample",
                temperature=1.0,
                top_k=None,
                top_p=0.9,
                top_p_include_boundary=False,
                seed=args.seed + point_index * 100_000 + batch_index,
                num_steps=64,
                steps_per_block=steps,
                first_hitting=first_hitting,
                use_kv_cache=True,
                start_token_id=50256,
            )
            precision_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else contextlib.nullcontext()
            )
            with precision_context:
                result = generate_fast_bd3(model, generation)
            batches.append(result.tokens.cpu())
            produced += size
            elapsed = time.perf_counter() - stage_start
            print(
                f"[sampling] point={point_id} batch={batch_index + 1}/{n_batches} "
                f"samples={produced}/{args.n_samples} elapsed={elapsed:.1f}s "
                f"samples_per_s={produced / max(elapsed, 1e-9):.2f}", flush=True,
            )
        tokens = torch.cat(batches, dim=0)[: args.n_samples]
        total_nfe = 256 if first_hitting else 16 * steps
        points.append({
            "id": point_id,
            "nfe": 256 if first_hitting else steps,
            "steps_per_block": 16 if first_hitting else steps,
            "configured_steps_per_block": steps,
            "total_denoising_nfe": total_nfe,
            "discretize": "sample",
            "forwards_per_sequence": total_nfe,
            "backbone_calls_per_sequence": total_nfe,
            "n_samples": int(tokens.shape[0]),
            "sampling_seconds": round(time.perf_counter() - stage_start, 1),
            "texts": tokenizer.batch_decode(tokens, skip_special_tokens=True),
        })
        token_arrays[point_id] = tokens.numpy().astype(np.int32)

        payload_out = {
            "schema": SCHEMA,
            "label": args.label,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": {
                "kind": "checkpoint",
                "checkpoint": str(checkpoint),
                "checkpoint_step": payload["step"],
                "checkpoint_sha256": _sha256(checkpoint),
                "dataset": "tinystories",
                "tokenizer": "gpt2",
                "arch": payload["config"]["model"]["backbone"],
            },
            "sampler": {
                "kind": "bd3lm_fused_fast",
                "block_size": model.block_size,
                "length": 256,
                "seed": args.seed,
                "cache_commit": "fused_with_next_first_step",
                "separate_cache_calls": 0,
                "precision": args.precision,
                "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
                "compile_mode": args.compile_mode,
            },
            "points": points,
        }
        out.write_text(json.dumps(payload_out, indent=1))
        np.savez_compressed(out.with_suffix(".tokens.npz"), **token_arrays)
        print(
            f"[artifact] point={point_id} json={out} tokens={out.with_suffix('.tokens.npz')}",
            flush=True,
        )
        torch.cuda.empty_cache()

    print(
        f"[done] points={len(points)} elapsed={time.perf_counter() - total_start:.1f}s "
        f"json={out} tokens={out.with_suffix('.tokens.npz')}", flush=True,
    )


if __name__ == "__main__":
    main()
