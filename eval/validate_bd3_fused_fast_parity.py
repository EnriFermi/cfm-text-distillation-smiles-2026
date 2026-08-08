"""Validate exact token parity between canonical and fused BD3-LM samplers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _token_hash(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()


def _file_hash(path: Path) -> str:
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
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/tinystories_fused_sampler_h100_20260807/bd3_token_parity.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.baseline_code_root.resolve()))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.bd3lm_fast import generate_fast_bd3
    from bcfm_baselines.models.generative_text import GenerationConfig

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[config] checkpoint={args.checkpoint.resolve()} device={args.device} "
        f"dtype=float32 seed={args.seed} batch=1 length=256 output={args.out.resolve()}",
        flush=True,
    )
    print("[stage=model_load]", flush=True)
    model, payload = load_checkpoint_model(args.checkpoint.resolve(), args.device)
    if payload.get("step") != 100000 or payload["config"]["dataset"]["name"] != "tinystories":
        raise RuntimeError("unexpected BD3-LM checkpoint provenance")
    print(
        f"[stage=model_ready] gpu={torch.cuda.get_device_name(0)} "
        f"parameter_dtype={next(model.parameters()).dtype} block_size={model.block_size}",
        flush=True,
    )
    checkpoint_sha256 = _file_hash(args.checkpoint.resolve())
    source_hashes = {
        "bcfm_baselines/models/bd3lm.py": _file_hash(
            args.baseline_code_root.resolve() / "bcfm_baselines/models/bd3lm.py"
        ),
        "bcfm_baselines/models/bd3lm_fast.py": _file_hash(
            args.baseline_code_root.resolve() / "bcfm_baselines/models/bd3lm_fast.py"
        ),
    }

    specs = [(f"ancestral_s{steps}", steps, False) for steps in (1, 2, 4, 8, 16, 32)]
    specs.append(("first_hitting_nfe256", 256, True))
    rows: list[dict] = []
    started = time.perf_counter()
    for index, (point, steps, first_hitting) in enumerate(specs, 1):
        point_started = time.perf_counter()
        config = GenerationConfig(
            length=256,
            batch_size=1,
            strategy="sample",
            temperature=1.0,
            top_k=None,
            top_p=0.9,
            top_p_include_boundary=False,
            seed=args.seed,
            num_steps=64,
            steps_per_block=steps,
            first_hitting=first_hitting,
            use_kv_cache=True,
            start_token_id=50256,
        )
        print(
            f"[stage=parity] point={point} progress={index}/{len(specs)} "
            f"first_hitting={first_hitting}", flush=True,
        )
        canonical = model.generate(config)
        fused = generate_fast_bd3(model, config)
        equal = torch.equal(canonical.tokens, fused.tokens)
        agreement = float((canonical.tokens == fused.tokens).float().mean().item())
        row = {
            "point": point,
            "configured_steps_per_block": steps,
            "effective_steps_per_block": 16 if first_hitting else steps,
            "total_denoising_nfe": 256 if first_hitting else 16 * steps,
            "exact_token_equal": equal,
            "token_agreement": agreement,
            "canonical_token_sha256": _token_hash(canonical.tokens),
            "fused_token_sha256": _token_hash(fused.tokens),
            "canonical_nfe": canonical.diagnostics["num_function_evaluations"],
            "fused_nfe": fused.diagnostics["num_function_evaluations"],
            "seconds": time.perf_counter() - point_started,
        }
        rows.append(row)
        args.out.write_text(json.dumps({
            "metadata": {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_step": payload["step"],
                "checkpoint_sha256": checkpoint_sha256,
                "implementation_source_sha256": source_hashes,
                "device": args.device,
                "gpu_name": torch.cuda.get_device_name(0),
                "dtype": "float32",
                "seed": args.seed,
                "batch_size": 1,
                "length": 256,
            },
            "rows": rows,
        }, indent=2) + "\n")
        print(
            f"[metric] point={point} exact={equal} agreement={agreement:.6f} "
            f"nfe={row['fused_nfe']} elapsed={row['seconds']:.1f}s", flush=True,
        )
    if not all(row["exact_token_equal"] for row in rows):
        raise RuntimeError("BD3-LM fused sampler parity failed")
    print(
        f"[done] rows={len(rows)} elapsed={time.perf_counter() - started:.1f}s "
        f"artifact={args.out.resolve()}", flush=True,
    )


if __name__ == "__main__":
    main()
