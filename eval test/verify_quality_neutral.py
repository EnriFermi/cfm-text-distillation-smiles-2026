"""Checkpoint-level token-identity checks for latency optimizations.

This is deliberately stricter than a quality metric: an optimization is admitted
to the "quality-neutral" latency sweep only when the same checkpoint, sampler,
seed, and FP32 precision produce exactly the same token tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from common import BD3_CKPT, CFM_CKPT, RESULTS, bootstrap


SEEDS = (0, 1, 1234, 12345)
ANCESTRAL_STEPS = (1, 4, 16)
BATCH_SIZE = 2


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_cfm() -> list[dict]:
    bootstrap("cfm")
    import cfm_fast
    from block.sampling import block_causal_sample
    from eval.dump_samples import build_module, detect_arch

    checkpoint = torch.load(CFM_CKPT, map_location="cpu", weights_only=False)
    arch = detect_arch(checkpoint)
    arch["_path"] = str(CFM_CKPT)
    del checkpoint
    module = build_module(arch, "cuda")

    rows: list[dict] = []
    for steps in ANCESTRAL_STEPS:
        for seed in SEEDS:
            seed_all(seed)
            reference = block_causal_sample(
                module,
                block_size=arch["block_size"],
                steps_per_block=steps,
                batch_size=BATCH_SIZE,
                discretize="argmax",
            )
            seed_all(seed)
            optimized = cfm_fast.cached_block_causal_sample(
                module,
                arch["block_size"],
                steps,
                batch_size=BATCH_SIZE,
                discretize="argmax",
            )
            agreement = (reference == optimized).float().mean().item()
            row = {
                "model": "cfm",
                "optimization": "sparse_vocab_active_block_clean_kv_cache",
                "sampler": "block_causal_argmax",
                "steps_per_block": steps,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "token_agreement": agreement,
                "identical": bool(torch.equal(reference, optimized)),
            }
            rows.append(row)
            print(
                f"[verify] cfm steps={steps:2d} seed={seed:5d} "
                f"agreement={agreement:.6f} identical={row['identical']}"
            )

    del module
    torch.cuda.empty_cache()
    return rows


def bd3_configs(base_sampling: dict) -> list[tuple[str, object]]:
    from bcfm_baselines.models.generative_text import GenerationConfig

    configs: list[tuple[str, GenerationConfig]] = []
    for steps in ANCESTRAL_STEPS:
        for seed in SEEDS:
            configs.append(
                (
                    "ancestral",
                    GenerationConfig(
                        **{
                            **base_sampling,
                            "batch_size": BATCH_SIZE,
                            "steps_per_block": steps,
                            "first_hitting": False,
                            "seed": seed,
                        }
                    ),
                )
            )
    for seed in SEEDS:
        configs.append(
            (
                "first_hitting",
                GenerationConfig(
                    **{
                        **base_sampling,
                        "batch_size": BATCH_SIZE,
                        "steps_per_block": 16,
                        "first_hitting": True,
                        "seed": seed,
                    }
                ),
            )
        )
    return configs


def verify_bd3() -> list[dict]:
    bootstrap("bd3lm")
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.generative_text import GenerationConfig

    model, payload = load_checkpoint_model(BD3_CKPT, "cuda")
    configs = bd3_configs(dict(payload["config"]["sampling"]))
    rows: list[dict] = []
    cached_references: dict[tuple[str, int, int], torch.Tensor] = {}

    for sampler, config in configs:
        assert isinstance(config, GenerationConfig)
        uncached_config = GenerationConfig(**{**config.__dict__, "use_kv_cache": False})
        cached_config = GenerationConfig(**{**config.__dict__, "use_kv_cache": True})
        uncached = model.generate(uncached_config).tokens
        cached = model.generate(cached_config).tokens
        agreement = (uncached == cached).float().mean().item()
        key = (sampler, int(config.steps_per_block or config.num_steps), config.seed)
        cached_references[key] = cached.cpu()
        row = {
            "model": "bd3lm",
            "optimization": "kv_cache",
            "sampler": sampler,
            "steps_per_block": key[1],
            "batch_size": BATCH_SIZE,
            "seed": config.seed,
            "token_agreement": agreement,
            "identical": bool(torch.equal(uncached, cached)),
        }
        rows.append(row)
        print(
            f"[verify] bd3 cache sampler={sampler:13s} steps={key[1]:2d} "
            f"seed={config.seed:5d} agreement={agreement:.6f} "
            f"identical={row['identical']}"
        )

    model.backbone = torch.compile(model.backbone, dynamic=True)
    for sampler, config in configs:
        assert isinstance(config, GenerationConfig)
        cached_config = GenerationConfig(**{**config.__dict__, "use_kv_cache": True})
        compiled = model.generate(cached_config).tokens.cpu()
        key = (sampler, int(config.steps_per_block or config.num_steps), config.seed)
        reference = cached_references[key]
        agreement = (reference == compiled).float().mean().item()
        row = {
            "model": "bd3lm",
            "optimization": "torch_compile_on_cached_backbone",
            "sampler": sampler,
            "steps_per_block": key[1],
            "batch_size": BATCH_SIZE,
            "seed": config.seed,
            "token_agreement": agreement,
            "identical": bool(torch.equal(reference, compiled)),
        }
        rows.append(row)
        print(
            f"[verify] bd3 compile sampler={sampler:13s} steps={key[1]:2d} "
            f"seed={config.seed:5d} agreement={agreement:.6f} "
            f"identical={row['identical']}"
        )

    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    rows = verify_cfm() + verify_bd3()
    summary: dict[str, dict] = {}
    for model, optimization in sorted({(r["model"], r["optimization"]) for r in rows}):
        selected = [
            r for r in rows
            if r["model"] == model and r["optimization"] == optimization
        ]
        summary[f"{model}:{optimization}"] = {
            "checks": len(selected),
            "all_identical": all(r["identical"] for r in selected),
            "minimum_token_agreement": min(r["token_agreement"] for r in selected),
        }

    output = {
        "meta": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "fp32",
            "seeds": list(SEEDS),
            "ancestral_steps": list(ANCESTRAL_STEPS),
            "batch_size": BATCH_SIZE,
            "criterion": "torch.equal on final token tensors",
        },
        "summary": summary,
        "rows": rows,
    }
    path = RESULTS / "quality_neutral_verification_20260805.json"
    path.write_text(json.dumps(output, indent=1) + "\n")
    print(f"[saved] {path}")
    if not all(item["all_identical"] for item in summary.values()):
        raise SystemExit("one or more quality-neutral candidates failed token identity")


if __name__ == "__main__":
    main()
