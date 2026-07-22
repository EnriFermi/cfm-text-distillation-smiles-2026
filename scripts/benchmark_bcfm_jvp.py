#!/usr/bin/env python3
"""Benchmark the sparse custom-kernel JVP used by CFM/ECLD.

The benchmark uses the real Text8 attention geometry and differentiates through
the JVP, as training does. It compares SemiCat's Triton kernel with the exact
PyTorch math-SDPA fallback. No model or optimizer state is changed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from semicat.jvp_utils.functional import safe_sdpa_jvp


def _block_attention(
    query: torch.Tensor,
    noisy_key: torch.Tensor,
    noisy_value: torch.Tensor,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    *,
    block_size: int,
    backend: str,
) -> torch.Tensor:
    outputs = []
    for lo in range(0, query.shape[1], block_size):
        hi = lo + block_size
        q_block = query[:, lo:hi].contiguous()
        k_block = torch.cat(
            (clean_key[:, :lo], noisy_key[:, lo:hi]),
            dim=1,
        ).contiguous()
        v_block = torch.cat(
            (clean_value[:, :lo], noisy_value[:, lo:hi]),
            dim=1,
        ).contiguous()
        if backend == "triton":
            output = safe_sdpa_jvp(q_block, k_block, v_block)
        elif backend == "math":
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.MATH
            ):
                output = F.scaled_dot_product_attention(
                    q_block.transpose(1, 2),
                    k_block.transpose(1, 2),
                    v_block.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
        else:
            raise ValueError(backend)
        outputs.append(output)
    return torch.cat(outputs, dim=1)


def _measure(step, repeats: int) -> dict[str, float]:
    step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return {
        "median_step_ms": statistics.median(timings),
        "min_step_ms": min(timings),
        "max_step_ms": max(timings),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(12345)
    batch = args.batch_size
    length = 256
    block_size = 16
    heads = 12
    head_dim = 64
    shape = (batch, length, heads, head_dim)
    dtype = torch.bfloat16
    print(
        "[jvp-bench] stage=setup "
        f"device={torch.cuda.get_device_name()} dtype={dtype} shape={shape}"
    )

    primals = tuple(
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    )
    tangents = tuple(torch.randn_like(tensor) for tensor in primals)
    clean_key = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    clean_value = torch.randn_like(clean_key, requires_grad=True)
    leaves = (*primals, clean_key, clean_value)

    def evaluate(backend: str):
        return torch.func.jvp(
            lambda q, k, v: _block_attention(
                q,
                k,
                v,
                clean_key,
                clean_value,
                block_size=block_size,
                backend=backend,
            ),
            primals,
            tangents,
        )

    print("[jvp-bench] stage=numerics backends=triton,math")
    triton_primal, triton_tangent = evaluate("triton")
    math_primal, math_tangent = evaluate("math")
    primal_difference = (triton_primal - math_primal).abs().float()
    tangent_difference = (triton_tangent - math_tangent).abs().float()
    del triton_primal, triton_tangent, math_primal, math_tangent

    def make_step(backend: str):
        def step() -> None:
            for tensor in leaves:
                tensor.grad = None
            primal, tangent = evaluate(backend)
            (primal.float().square().mean() + tangent.float().square().mean()).backward()

        return step

    results = {}
    for backend in ("triton", "math"):
        print(f"[jvp-bench] stage=measure backend={backend}")
        results[backend] = _measure(make_step(backend), args.repeats)

    report = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "batch_size": batch,
        "length": length,
        "block_size": block_size,
        "heads": heads,
        "head_dim": head_dim,
        "repeats": args.repeats,
        "results": results,
        "triton_speedup_over_math": (
            results["math"]["median_step_ms"]
            / results["triton"]["median_step_ms"]
        ),
        "primal_max_abs": float(primal_difference.max()),
        "primal_mean_abs": float(primal_difference.mean()),
        "tangent_max_abs": float(tangent_difference.max()),
        "tangent_mean_abs": float(tangent_difference.mean()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[jvp-bench] stage=write artifact={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
