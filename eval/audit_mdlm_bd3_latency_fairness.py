"""Targeted multi-seed audit of the optimized MDLM/BD3-LM latency paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timed(call, repeats: int) -> tuple[list[float], object]:
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    assert result is not None
    return samples, result


def _summary(rows: list[dict], latency_key: str = "latency_ms") -> dict:
    values = [float(row[latency_key]) for row in rows]
    return {
        "seeds": len(rows),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-checkpoint", type=Path, default=ROOT / "baseline/MDLM_best.pt")
    parser.add_argument("--bd3-checkpoint", type=Path, default=ROOT / "baseline/BD3LM_best.pt")
    parser.add_argument(
        "--baseline-code-root", type=Path,
        default=ROOT / "external/bcfm_baselines_uploaded_20260807",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    os.environ["BCFM_INFERENCE_COMPILE_MODE"] = "max-autotune"
    code_root = args.baseline_code_root.resolve()
    sys.path.insert(0, str(code_root))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.bd3lm_fast import generate_fast_bd3
    from bcfm_baselines.models.generative_text import GenerationConfig
    from bcfm_baselines.models.mdlm_fast import generate_fast_mdlm

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "mdlm_bd3_latency_fairness_audit",
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bf16",
        "compile_mode": "max-autotune",
        "batch_size": 1,
        "length": 256,
        "seeds": args.seeds,
        "repeats_per_seed": args.repeats,
        "mdlm_checkpoint": str(args.mdlm_checkpoint.resolve()),
        "mdlm_checkpoint_sha256": _sha256(args.mdlm_checkpoint.resolve()),
        "bd3_checkpoint": str(args.bd3_checkpoint.resolve()),
        "bd3_checkpoint_sha256": _sha256(args.bd3_checkpoint.resolve()),
        "output": str(output),
    }
    print("[config] " + json.dumps(metadata, indent=2), flush=True)
    payload: dict = {"metadata": metadata, "mdlm": {}, "bd3lm": {}}

    print("[stage=model_load] model=MDLM", flush=True)
    mdlm, mdlm_checkpoint = load_checkpoint_model(args.mdlm_checkpoint.resolve(), "cuda")
    mdlm = mdlm.to(dtype=torch.bfloat16).eval()
    if mdlm_checkpoint.get("weights_used_for_inference") != "ema":
        raise RuntimeError("MDLM did not load EMA weights")

    def mdlm_config(nfe: int, seed: int, cache: bool) -> GenerationConfig:
        return GenerationConfig(
            length=256, batch_size=1, strategy="sample", temperature=1.0,
            top_k=None, top_p=0.9, top_p_include_boundary=False, seed=seed,
            num_steps=nfe, sampling_epsilon=1e-5, noise_removal=True,
            cache_denoiser_outputs=cache, start_token_id=50256,
        )

    print("[stage=compile_warmup] model=MDLM", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generate_fast_mdlm(mdlm, mdlm_config(16, args.seeds[0], True))
        generate_fast_mdlm(mdlm, mdlm_config(16, args.seeds[0], True))
    torch.cuda.synchronize()

    mdlm_rows: list[dict] = []
    mdlm_nfes = (16, 32, 64, 128, 256)
    for nfe in mdlm_nfes:
        for seed in args.seeds:
            for cache in (True, False):
                def call(nfe=nfe, seed=seed, cache=cache):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return generate_fast_mdlm(mdlm, mdlm_config(nfe, seed, cache))

                samples, result = _timed(call, args.repeats)
                row = {
                    "nfe": nfe,
                    "seed": seed,
                    "cache": cache,
                    "latency_ms": statistics.median(samples),
                    "latency_samples_ms": samples,
                    "actual_backbone_calls": result.diagnostics["num_function_evaluations"],
                    "cache_hits": result.diagnostics["denoiser_cache_hits"],
                }
                mdlm_rows.append(row)
                print(
                    f"[metric] model=MDLM nfe={nfe} seed={seed} cache={cache} "
                    f"latency_ms={row['latency_ms']:.3f} calls={row['actual_backbone_calls']} "
                    f"hits={row['cache_hits']}", flush=True,
                )
    payload["mdlm"]["rows"] = mdlm_rows
    payload["mdlm"]["summary"] = {
        f"nfe{nfe}_{'cache' if cache else 'no_cache'}": {
            **_summary([
                row for row in mdlm_rows if row["nfe"] == nfe and row["cache"] == cache
            ]),
            "actual_calls": [
                row["actual_backbone_calls"] for row in mdlm_rows
                if row["nfe"] == nfe and row["cache"] == cache
            ],
            "cache_hits": [
                row["cache_hits"] for row in mdlm_rows
                if row["nfe"] == nfe and row["cache"] == cache
            ],
        }
        for nfe in mdlm_nfes for cache in (True, False)
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print("[stage=model_load] model=BD3-LM", flush=True)
    bd3, bd3_checkpoint = load_checkpoint_model(args.bd3_checkpoint.resolve(), "cuda")
    bd3 = bd3.to(dtype=torch.bfloat16).eval()
    if bd3_checkpoint.get("weights_used_for_inference") != "ema":
        raise RuntimeError("BD3-LM did not load EMA weights")

    def bd3_config(steps: int, seed: int) -> GenerationConfig:
        return GenerationConfig(
            length=256, batch_size=1, strategy="sample", temperature=1.0,
            top_k=None, top_p=0.9, top_p_include_boundary=False, seed=seed,
            num_steps=64, steps_per_block=steps, first_hitting=False,
            use_kv_cache=True, start_token_id=50256,
        )

    print("[stage=compile_warmup] model=BD3-LM", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generate_fast_bd3(bd3, bd3_config(1, args.seeds[0]))
        generate_fast_bd3(bd3, bd3_config(1, args.seeds[0]))
    torch.cuda.synchronize()

    bd3_rows: list[dict] = []
    for steps in (1, 2, 4, 8):
        # The first ancestral step uses the fused transition, while later steps
        # use the active scorer.  Warm every step-count once so S=1 cannot leave
        # unseen non-empty-prefix active shapes to compile inside the S=2 timing.
        print(f"[stage=point_warmup] model=BD3-LM steps={steps}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                generate_fast_bd3(bd3, bd3_config(steps, args.seeds[0]))
        torch.cuda.synchronize()
        for seed in args.seeds:
            # Warm the exact stochastic trajectory once. This makes the audit
            # estimate sequence-dependent runtime rather than first-use setup.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generate_fast_bd3(bd3, bd3_config(steps, seed))
            torch.cuda.synchronize()
            def call(steps=steps, seed=seed):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return generate_fast_bd3(bd3, bd3_config(steps, seed))

            samples, result = _timed(call, args.repeats)
            row = {
                "steps_per_block": steps,
                "seed": seed,
                "latency_ms": statistics.median(samples),
                "latency_samples_ms": samples,
                "actual_backbone_calls": result.diagnostics["num_function_evaluations"],
            }
            bd3_rows.append(row)
            print(
                f"[metric] model=BD3-LM steps={steps} seed={seed} "
                f"latency_ms={row['latency_ms']:.3f} calls={row['actual_backbone_calls']}",
                flush=True,
            )
    payload["bd3lm"]["rows"] = bd3_rows
    payload["bd3lm"]["summary"] = {
        f"steps{steps}": {
            **_summary([row for row in bd3_rows if row["steps_per_block"] == steps]),
            "actual_calls": [
                row["actual_backbone_calls"] for row in bd3_rows
                if row["steps_per_block"] == steps
            ],
        }
        for steps in (1, 2, 4, 8)
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[done] artifact={output}", flush=True)


if __name__ == "__main__":
    main()
