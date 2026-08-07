"""Generation latency of the TinyStories BD3-LM checkpoint (2 models/bd3lm/last.pt).

Sweeps the only two knobs that move inference cost — ``steps_per_block`` and the
sampler (fixed-grid ancestral vs. the paper's first-hitting one) — at the block
size baked into the checkpoint. ``model.generate`` is the same call the repo's
``sample``/``evaluate`` entry points make.

    python "eval test/bench_bd3lm.py" --steps-per-block 1 2 4 8 16 --batch-size 1 8 32
"""

from __future__ import annotations

import argparse
import contextlib

import torch

from common import (
    BD3_CKPT,
    bootstrap,
    gpu_name,
    peak_mem_gb,
    row,
    save,
    timed,
)

bootstrap("bd3lm")

import bd3_fast  # noqa: E402
from bcfm_baselines.checkpoint import load_checkpoint_model  # noqa: E402
from bcfm_baselines.models.generative_text import GenerationConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-block", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--first-hitting", action="store_true",
                        help="also time the paper's first-hitting sampler (needs S >= B)")
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--amp", choices=["bf16", "fp16"], default=None,
                        help="autocast the whole generate() call")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the backbone (dynamic: the cache grows)")
    parser.add_argument("--lean", action="store_true",
                        help="skip optional diagnostics without changing sampled tokens")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=str(BD3_CKPT))
    parser.add_argument("--out", default="bd3lm_latency.json")
    args = parser.parse_args()
    if args.amp is None:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model, payload = load_checkpoint_model(args.checkpoint, args.device)
    config = payload["config"]
    base_sampling = dict(config["sampling"])
    block_size = model.block_size
    length = base_sampling.get("length", 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bd3lm] block_size={block_size} params={n_params:,} length={length} "
          f"weights={payload['weights_used_for_inference']} "
          f"step={payload.get('step', payload.get('global_step'))}")
    print(f"[gpu] {gpu_name()}\n")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    if args.compile:
        # The KV cache grows a block per step, so the backbone sees 16 different
        # key lengths; dynamic=True keeps that to one graph instead of sixteen.
        model.backbone = torch.compile(model.backbone, dynamic=True)
    if args.amp or args.compile:
        opts = ([args.amp] if args.amp else []) + (["torch.compile"] if args.compile else [])
        print(f"[opt] {', '.join(opts)}\n")

    samplers = [("ancestral", False)] + ([("first_hitting", True)] if args.first_hitting else [])
    rows = []
    for batch_size in args.batch_size:
        for sampler_name, first_hitting in samplers:
            for steps in args.steps_per_block:
                # The first-hitting sampler reveals exactly one token per NFE, so
                # it cannot finish a block in fewer than block_size steps.
                if first_hitting and steps < block_size:
                    continue
                generation = GenerationConfig(**{
                    **base_sampling,
                    "batch_size": batch_size,
                    "length": length,
                    "steps_per_block": steps,
                    "first_hitting": first_hitting,
                    "use_kv_cache": not args.no_kv_cache,
                })
                torch.cuda.reset_peak_memory_stats()
                result_holder = {}

                def once(generation=generation):
                    with torch.inference_mode():
                        with (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype
                              else contextlib.nullcontext()):
                            result_holder["r"] = (
                                bd3_fast.generate(model, generation)
                                if args.lean else model.generate(generation)
                            )

                try:
                    stats = timed(once, warmup=args.warmup, repeats=args.repeats)
                except torch.cuda.OutOfMemoryError:
                    print(f"  batch={batch_size:3d} {sampler_name:13s} S={steps:3d}  OOM")
                    torch.cuda.empty_cache()
                    continue

                nfe = result_holder["r"].diagnostics["num_function_evaluations"]
                entry = row(model="bd3lm", batch_size=batch_size, length=length,
                            stats=stats, sampler=sampler_name, steps_per_block=steps,
                            block_size=block_size, nfe_per_sequence=nfe,
                            # each block also costs one KV-cache append forward
                            forwards_per_sequence=nfe + length // block_size,
                            kv_cache=not args.no_kv_cache,
                            diagnostics=not args.lean,
                            amp=args.amp, compiled=args.compile,
                            peak_mem_gb=round(peak_mem_gb(), 2))
                rows.append(entry)
                print(f"  batch={batch_size:3d} {sampler_name:13s} S={steps:3d}  "
                      f"{entry['latency_s_per_batch']:8.4f} s/batch  "
                      f"{entry['latency_ms_per_sequence']:9.2f} ms/seq  "
                      f"{entry['tokens_per_sec']:10.1f} tok/s  "
                      f"NFE={nfe:4d}  peak {entry['peak_mem_gb']:.2f} GB")
                torch.cuda.empty_cache()

    save(rows, args.out, {
        "model": "bd3lm", "checkpoint": args.checkpoint,
        "weights_used_for_inference": payload["weights_used_for_inference"],
        "block_size": block_size, "n_params": n_params, "length": length,
        "gpu": gpu_name(), "device": args.device, "torch": torch.__version__,
        "dtype": args.amp or "fp32",
        "precision": config.get("training", {}).get("precision"),
        "warmup": args.warmup, "repeats": args.repeats,
        "base_sampling": base_sampling,
        "diagnostics": not args.lean,
        "tf32": False if args.amp is None else None,
    })


if __name__ == "__main__":
    main()
