"""Low-NFE H100 discriminator for literal preallocated vs dynamic fused KV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from block.fast_inference import (
    compiled_fused_transition_step,
    compiled_inplace_flow_step,
)
from block.sampling import _discretize, uniform_schedule
from eval.dump_samples import build_module, detect_arch, sample_point
from eval.measure_fused_sampler_h100_latency import _load_upstream_timer


ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def _sample_literal_preallocated(module, *, steps_per_block: int = 1) -> torch.Tensor:
    block_size = int(module.net.block_size)
    length, vocab = int(module.in_shape[0]), int(module.in_shape[-1])
    batch_size = 1
    device = module.device
    dtype = next(module.net.parameters()).dtype
    head_dim = module.net.hidden_size // module.net.n_heads
    storage_shape = (batch_size, length, module.net.n_heads, head_dim)
    key_storage = tuple(
        torch.empty(storage_shape, device=device, dtype=dtype)
        for _ in module.net.blocks
    )
    value_storage = tuple(torch.empty_like(item) for item in key_storage)
    flow_step = compiled_inplace_flow_step(module.net)
    transition_step = compiled_fused_transition_step(module.net)
    schedule = uniform_schedule(steps_per_block)
    tokens = torch.empty(batch_size, length, dtype=torch.long, device=device)
    previous_clean = None

    for block_index in range(length // block_size):
        lo = block_index * block_size
        hi = lo + block_size
        positions = torch.arange(lo, hi, device=device)
        noisy = module.prior((batch_size, block_size, vocab), device=device)
        for step_index, (s_value, t_value) in enumerate(schedule):
            s = torch.full(
                (batch_size, block_size), s_value, device=device, dtype=noisy.dtype,
            )
            t = torch.full_like(s, t_value)
            step_scale = torch.scalar_tensor(
                (t_value - s_value) / (1.0 - s_value + 1e-8),
                device=device, dtype=noisy.dtype,
            )
            if block_index and step_index == 0:
                if previous_clean is None:
                    raise RuntimeError("missing finalized previous block")
                previous_positions = torch.arange(lo - block_size, lo, device=device)
                noisy = transition_step(
                    previous_clean, noisy, s, t, previous_positions, positions,
                    lo - block_size, step_scale, key_storage, value_storage,
                )
            else:
                noisy = flow_step(
                    noisy, s, t, positions, lo, step_scale,
                    key_storage, value_storage,
                )
            noisy = noisy.clone()
        block = _discretize(noisy, "argmax")
        tokens[:, lo:hi] = block
        if block_index + 1 < length // block_size:
            previous_clean = F.one_hot(block, vocab).to(noisy.dtype)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt",
    )
    parser.add_argument(
        "--upstream-timing-source", type=Path,
        default=Path("/tmp/cfm-upstream-m2-latency-2e21c57/eval/run_eval.py"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "artifacts/tinystories_fused_sampler_h100_20260807/cache_backend_ablation.json",
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[config] experiment=cache_backend_ablation model=M3 steps_per_block=1 "
        f"device={args.device} dtype=float32 seed={args.seed} warmup={args.warmup_runs} "
        f"repeats={args.repeats} output={args.out.resolve()}", flush=True,
    )
    print(f"[stage=model_load] checkpoint={args.checkpoint.resolve()}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = detect_arch(checkpoint)
    arch["_path"] = args.checkpoint.resolve()
    del checkpoint
    module = build_module(arch, args.device)
    print(
        f"[stage=model_ready] gpu={torch.cuda.get_device_name(0)} "
        f"parameter_dtype={next(module.parameters()).dtype}", flush=True,
    )

    backend = "dynamic"

    def sample_tokens(_module, _sampler, *, n_samples, batch_size, length):
        if (n_samples, batch_size, length) != (1, 1, 256):
            raise ValueError("ablation protocol requires n=1, batch=1, length=256")
        if backend == "preallocated":
            return _sample_literal_preallocated(_module, steps_per_block=1)
        return sample_point(
            _module, "block_causal_fused_fast", arch, 1, "argmax", 1, 1,
        )

    timer = _load_upstream_timer(args.upstream_timing_source, sample_tokens)
    sampler = OmegaConf.create({"kind": "cache_ablation"})
    results: dict[str, dict] = {}
    started = time.perf_counter()
    for name in ("dynamic", "preallocated"):
        backend = name
        print(f"[stage=latency] backend={name}", flush=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        results[name] = timer(
            module, sampler, length=256, device=args.device,
            warmup_runs=args.warmup_runs, repeats=args.repeats,
        )
        print(
            f"[metric] backend={name} median_ms={results[name]['sequence_latency_ms']:.3f} "
            f"p10={results[name]['sequence_latency_p10_ms']:.3f} "
            f"p90={results[name]['sequence_latency_p90_ms']:.3f}", flush=True,
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dynamic_tokens = sample_point(
        module, "block_causal_fused_fast", arch, 1, "argmax", 1, 1,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    preallocated_tokens = _sample_literal_preallocated(module, steps_per_block=1)
    artifact = {
        "metadata": {
            "experiment": "M3 NFE/block=1 cache backend discriminator",
            "checkpoint": str(args.checkpoint.resolve()),
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": "float32",
            "seed": args.seed,
            "batch_size": 1,
            "length": 256,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "upstream_timing_source": str(args.upstream_timing_source.resolve()),
        },
        "results": results,
        "preallocated_over_dynamic_slowdown": (
            results["preallocated"]["sequence_latency_ms"]
            / results["dynamic"]["sequence_latency_ms"]
        ),
        "exact_token_equal": torch.equal(
            dynamic_tokens.detach().cpu(), preallocated_tokens.detach().cpu(),
        ),
        "token_agreement": float(
            (
                dynamic_tokens.detach().cpu()
                == preallocated_tokens.detach().cpu()
            ).float().mean().item()
        ),
    }
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        f"[done] slowdown={artifact['preallocated_over_dynamic_slowdown']:.3f}x "
        f"token_agreement={artifact['token_agreement']:.6f} "
        f"elapsed={time.perf_counter() - started:.1f}s artifact={args.out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
