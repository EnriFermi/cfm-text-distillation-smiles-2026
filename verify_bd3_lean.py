"""Verify that the eval-only lean BD3 generator preserves sampled tokens."""

from __future__ import annotations

import json

import torch

from common import BD3_CKPT, RESULTS, bootstrap


def main() -> None:
    bootstrap("bd3lm")
    import bd3_fast
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.generative_text import GenerationConfig

    model, payload = load_checkpoint_model(BD3_CKPT, "cuda")
    base = dict(payload["config"]["sampling"])
    rows = []
    points = [("ancestral", s, False) for s in (1, 4, 16)]
    points.append(("first_hitting", 16, True))
    for sampler, steps, first_hitting in points:
        for seed in (0, 1, 1234, 12345):
            config = GenerationConfig(
                **{
                    **base,
                    "batch_size": 2,
                    "steps_per_block": steps,
                    "first_hitting": first_hitting,
                    "use_kv_cache": True,
                    "seed": seed,
                }
            )
            reference = model.generate(config).tokens
            lean = bd3_fast.generate(model, config).tokens
            identical = torch.equal(reference, lean)
            agreement = (reference == lean).float().mean().item()
            rows.append(
                {
                    "sampler": sampler,
                    "steps_per_block": steps,
                    "seed": seed,
                    "token_agreement": agreement,
                    "identical": identical,
                }
            )
            print(
                f"[lean] sampler={sampler:13s} steps={steps:2d} seed={seed:5d} "
                f"agreement={agreement:.6f} identical={identical}"
            )
    output = {
        "meta": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "fp32",
            "checkpoint": str(BD3_CKPT),
            "criterion": "torch.equal on final token tensors",
        },
        "all_identical": all(r["identical"] for r in rows),
        "rows": rows,
    }
    path = RESULTS / "bd3_lean_verification_20260805.json"
    path.write_text(json.dumps(output, indent=1) + "\n")
    print(f"[saved] {path}")
    if not output["all_identical"]:
        raise SystemExit("lean BD3 generator changed sampled tokens")


if __name__ == "__main__":
    main()
