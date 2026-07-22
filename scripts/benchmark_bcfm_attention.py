#!/usr/bin/env python3
"""Benchmark BCFM's production doubled-sequence attention on CUDA.

This is an operator benchmark, not a training run: no checkpoint is loaded and
no optimizer step is performed. It compares FlexAttention mask tile sizes with
the dense SDPA fallback at the actual Text8 sequence/head geometry.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from block.block_dit import _compiled_flex_attention
from block.mask import block_causal_mask, create_block_causal_flex_mask


def _measure(step, *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    timings: list[float] = []
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    length = 256
    block_size = 16
    heads = 12
    head_dim = 64
    tokens = 2 * length
    dtype = torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(12345)
    print(
        "[attention-bench] stage=setup "
        f"device={torch.cuda.get_device_name()} dtype={dtype} "
        f"batch={args.batch_size} heads={heads} tokens={tokens} head_dim={head_dim}"
    )

    query = torch.randn(
        args.batch_size,
        heads,
        tokens,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)

    def clear_gradients() -> None:
        query.grad = None
        key.grad = None
        value.grad = None

    results: dict[str, dict[str, object]] = {}
    masks = {}
    for kernel_block_size in (16, 32, 64, 128):
        print(
            "[attention-bench] stage=mask "
            f"backend=flex kernel_block_size={kernel_block_size}"
        )
        mask = create_block_causal_flex_mask(
            length,
            block_size,
            device,
            kernel_block_size=kernel_block_size,
        )
        masks[kernel_block_size] = mask

        def flex_step(mask=mask) -> None:
            clear_gradients()
            output = _compiled_flex_attention(query, key, value, mask)
            output.float().square().mean().backward()

        print(
            "[attention-bench] stage=measure "
            f"backend=flex kernel_block_size={kernel_block_size}"
        )
        name = f"flex_tile_{kernel_block_size}"
        try:
            measurement = _measure(
                flex_step,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results[name] = {"status": "pass", **measurement}
        except Exception as error:
            # FlexAttention supports mask metadata tiles more broadly than its
            # compiled CUDA kernels do. Keep unsupported candidates in the
            # report instead of aborting the benchmark.
            results[name] = {
                "status": "unsupported",
                "error_type": type(error).__name__,
                "error": str(error).splitlines()[0],
            }
            print(
                "[attention-bench] stage=unsupported "
                f"backend=flex kernel_block_size={kernel_block_size} "
                f"error={type(error).__name__}: {str(error).splitlines()[0]}"
            )
            torch.cuda.empty_cache()

    dense_mask = block_causal_mask(length, block_size, device)

    def dense_step() -> None:
        clear_gradients()
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=dense_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output.float().square().mean().backward()

    print("[attention-bench] stage=measure backend=dense_sdpa")
    results["dense_sdpa"] = {
        "status": "pass",
        **_measure(
            dense_step,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }

    selected = min(
        (
            name
            for name, measurement in results.items()
            if name.startswith("flex_tile_") and measurement["status"] == "pass"
        ),
        key=lambda name: float(results[name]["median_step_ms"]),
    )
    selected_tile = int(selected.removeprefix("flex_tile_"))

    # Compare the selected production tile against exact math SDPA on one item.
    with torch.no_grad():
        q1, k1, v1 = query[:1], key[:1], value[:1]
        flex_output = _compiled_flex_attention(
            q1,
            k1,
            v1,
            masks[selected_tile],
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            dense_output = F.scaled_dot_product_attention(
                q1,
                k1,
                v1,
                attn_mask=dense_mask,
                dropout_p=0.0,
                is_causal=False,
            )
    difference = (flex_output - dense_output).abs().float()
    report = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "length": length,
        "doubled_tokens": tokens,
        "block_size": block_size,
        "heads": heads,
        "head_dim": head_dim,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
        "fastest_flex_tile": selected,
        "selected_tile_vs_math_max_abs": float(difference.max()),
        "selected_tile_vs_math_mean_abs": float(difference.mean()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[attention-bench] stage=write artifact={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
