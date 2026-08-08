"""Generate matched BF16/max-autotune TinyStories dumps with fast MDLM."""

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
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "baseline/MDLM_best.pt")
    parser.add_argument(
        "--baseline-code-root", type=Path,
        default=ROOT / "external/bcfm_baselines_uploaded_20260807",
    )
    parser.add_argument("--nfe", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=("fp32_tf32", "bf16"), default="bf16",
    )
    parser.add_argument(
        "--compile-mode", choices=("reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument("--label", default="MDLM_fused_fast_bf16_maxautotune")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    os.environ["BCFM_INFERENCE_COMPILE_MODE"] = args.compile_mode
    code_root = args.baseline_code_root.resolve()
    sys.path.insert(0, str(code_root))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.generative_text import GenerationConfig
    from bcfm_baselines.models.mdlm_fast import generate_fast_mdlm
    import transformers

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_hash = _sha256(checkpoint_path)
    print(
        f"[config] experiment=mdlm_fused_fast_quality device={args.device} "
        f"precision={args.precision} compile_mode={args.compile_mode} "
        f"seed={args.seed} nfe={args.nfe} n_samples={args.n_samples} "
        f"batch_size={args.batch_size} "
        f"cache_mode=denoiser_output_cache+compiled_cuda_graph output={out}",
        flush=True,
    )
    print(
        f"[stage=model_load] checkpoint={checkpoint_path} sha256={checkpoint_hash}",
        flush=True,
    )
    model, checkpoint = load_checkpoint_model(checkpoint_path, args.device)
    if checkpoint.get("step") != 100000:
        raise RuntimeError(f"unexpected MDLM checkpoint step {checkpoint.get('step')}")
    if checkpoint["config"]["dataset"]["name"] != "tinystories":
        raise RuntimeError("unexpected MDLM dataset")
    if checkpoint.get("weights_used_for_inference") != "ema":
        raise RuntimeError("MDLM did not load EMA weights")
    if args.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    print(
        f"[stage=model_ready] gpu={torch.cuda.get_device_name(0)} "
        f"parameter_dtype={next(model.parameters()).dtype} step={checkpoint['step']} "
        f"weights={checkpoint['weights_used_for_inference']}",
        flush=True,
    )

    points: list[dict] = []
    token_arrays: dict[str, np.ndarray] = {}
    total_start = time.perf_counter()
    for point_index, nfe in enumerate(args.nfe):
        point_id = f"sample_nfe{nfe}"
        stage_start = time.perf_counter()
        batches: list[torch.Tensor] = []
        diagnostics: list[dict] = []
        produced = 0
        n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
        print(
            f"[stage=sampling] point={point_id} progress={point_index + 1}/{len(args.nfe)} "
            f"batches={n_batches}",
            flush=True,
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
                num_steps=nfe,
                sampling_epsilon=1e-5,
                noise_removal=True,
                cache_denoiser_outputs=True,
                start_token_id=50256,
            )
            precision_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else contextlib.nullcontext()
            )
            with precision_context:
                result = generate_fast_mdlm(model, generation)
            batches.append(result.tokens.cpu())
            diagnostics.append(result.diagnostics)
            produced += size
            elapsed = time.perf_counter() - stage_start
            print(
                f"[sampling] point={point_id} batch={batch_index + 1}/{n_batches} "
                f"samples={produced}/{args.n_samples} "
                f"backbone_calls={result.diagnostics['num_function_evaluations']} "
                f"cache_hits={result.diagnostics['denoiser_cache_hits']} "
                f"elapsed={elapsed:.1f}s samples_per_s={produced / max(elapsed, 1e-9):.2f}",
                flush=True,
            )
        tokens = torch.cat(batches, dim=0)[: args.n_samples]
        observed_calls = [int(item["num_function_evaluations"]) for item in diagnostics]
        observed_hits = [int(item["denoiser_cache_hits"]) for item in diagnostics]
        points.append({
            "id": point_id,
            "nfe": nfe,
            "configured_diffusion_steps": nfe,
            "discretize": "sample",
            "forwards_per_sequence": nfe + 1,
            "configured_max_backbone_calls_per_sequence": nfe + 1,
            "observed_backbone_calls_min": min(observed_calls),
            "observed_backbone_calls_max": max(observed_calls),
            "observed_backbone_calls_mean": sum(observed_calls) / len(observed_calls),
            "observed_cache_hits_min": min(observed_hits),
            "observed_cache_hits_max": max(observed_hits),
            "observed_cache_hits_mean": sum(observed_hits) / len(observed_hits),
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
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": checkpoint["step"],
                "checkpoint_sha256": checkpoint_hash,
                "weights_used_for_inference": checkpoint["weights_used_for_inference"],
                "dataset": "tinystories",
                "tokenizer": "gpt2",
                "arch": checkpoint["config"]["model"]["backbone"],
            },
            "sampler": {
                "kind": "mdlm_fused_fast",
                "length": 256,
                "seed": args.seed,
                "strategy": "sample",
                "top_p": 0.9,
                "top_p_include_boundary": False,
                "sampling_epsilon": 1e-5,
                "noise_removal": True,
                "denoiser_output_cache": True,
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
        f"json={out} tokens={out.with_suffix('.tokens.npz')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
