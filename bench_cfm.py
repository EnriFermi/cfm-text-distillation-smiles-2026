"""Generation latency of the TinyStories CFM checkpoint (2 models/cfm_mod/last.ckpt).

Architecture is auto-detected from the tensors exactly the way the repo's own
``eval/dump_samples.py`` does it, so this benchmarks the same module the eval
harness would build — no separate config to drift.

    python "eval test/bench_cfm.py" --nfe 1 2 4 8 16 --batch-size 1 8 32
"""

from __future__ import annotations

import argparse

import torch

from common import (
    CFM_CKPT,
    bootstrap,
    gpu_name,
    peak_mem_gb,
    row,
    save,
    timed,
)

bootstrap("cfm")

import cfm_fast  # noqa: E402
from block.sampling import block_causal_sample  # noqa: E402
from eval.dump_samples import build_module, detect_arch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="flow-map jumps (per block for a block model)")
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=str(CFM_CKPT))
    parser.add_argument("--out", default="cfm_latency.json")
    parser.add_argument("--fast", action="store_true",
                        help="exact sparse-vocab sampler (see cfm_fast.py)")
    parser.add_argument("--cache", action="store_true",
                        help="also cache the clean-prefix KV across jumps (exact)")
    parser.add_argument("--amp", choices=["bf16", "fp16"], default=None)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the transformer body")
    parser.add_argument("--attn", choices=["flex", "sdpa"], default=None,
                        help="attention backend; 'sdpa' is the one --compile can "
                             "fuse (flex_attention is already compiled internally, "
                             "and nesting the two compilations fails)")
    args = parser.parse_args()
    if (args.amp or args.compile) and not args.fast:
        parser.error("--amp/--compile are only wired into the --fast sampler")
    if args.amp is None:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = detect_arch(checkpoint)
    arch["_path"] = args.checkpoint
    step = checkpoint.get("global_step")
    del checkpoint

    module = build_module(arch, args.device)
    n_params = sum(p.numel() for p in module.net.parameters())
    length = arch["length"]
    kind = "block_causal" if arch["block_size"] else "full_cfm"

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    body = None
    if args.fast:
        if kind != "block_causal":
            raise SystemExit("--fast is the block-causal sampler")
        cfm_fast.prepare(module.net)
        if args.attn:
            module.net.attention_backend = args.attn
        # Verified before any compile wrapper, so this compares the maths only:
        # the reference sampler runs flex, so agreement also proves the dense
        # SDPA mask is the same attention graph.
        check = cfm_fast.assert_matches_reference(module, arch["block_size"], 2,
                                                  cached=args.cache)
        print(f"[fast] exactness vs reference sampler (fp32): "
              f"token agreement {check['token_agreement']:.4f}")
        body = cfm_fast.make_body(module.net, compile_body=args.compile)
        opts = ["sparse-vocab"] + (["kv-cache"] if args.cache else []) + \
               ([args.amp] if args.amp else []) + \
               ([f"attn={args.attn}"] if args.attn else []) + \
               (["torch.compile"] if args.compile else [])
        print(f"[fast] optimizations: {', '.join(opts)}")
    print(f"[cfm] {kind} step={step} params={n_params:,} length={length} "
          f"vocab={arch['vocab_size']} hidden={arch['hidden_size']}x{arch['n_blocks']} "
          f"block_size={arch['block_size']}")
    print(f"[gpu] {gpu_name()}\n")

    rows = []
    for batch_size in args.batch_size:
        for nfe in args.nfe:
            torch.cuda.reset_peak_memory_stats()

            def once(batch_size=batch_size, nfe=nfe):
                torch.manual_seed(0)
                if args.cache:
                    return cfm_fast.cached_block_causal_sample(
                        module, arch["block_size"], nfe, batch_size=batch_size,
                        discretize="argmax", amp_dtype=amp_dtype)
                if args.fast:
                    return cfm_fast.fast_block_causal_sample(
                        module, arch["block_size"], nfe, batch_size=batch_size,
                        discretize="argmax", amp_dtype=amp_dtype, body=body)
                if kind == "full_cfm":
                    return module.sample_flow_map_batch(batch_size, nfe).argmax(-1)
                return block_causal_sample(
                    module, block_size=arch["block_size"], steps_per_block=nfe,
                    batch_size=batch_size, discretize="argmax")

            try:
                stats = timed(once, warmup=args.warmup, repeats=args.repeats)
            except torch.cuda.OutOfMemoryError:
                print(f"  batch={batch_size:3d} nfe={nfe:3d}  OOM")
                torch.cuda.empty_cache()
                continue

            # A full-sequence CFM does `nfe` forwards over the whole sequence; a
            # block-causal one does `nfe` per block.
            blocks = length // arch["block_size"] if arch["block_size"] else 1
            entry = row(model="cfm", batch_size=batch_size, length=length, stats=stats,
                        sampler=kind, nfe=nfe, block_size=arch["block_size"],
                        forwards_per_sequence=nfe * blocks,
                        fast=args.fast, kv_cache=args.cache, amp=args.amp,
                        compiled=args.compile,
                        peak_mem_gb=round(peak_mem_gb(), 2))
            rows.append(entry)
            print(f"  batch={batch_size:3d} nfe={nfe:3d}  "
                  f"{entry['latency_s_per_batch']:8.4f} s/batch  "
                  f"{entry['latency_ms_per_sequence']:9.2f} ms/seq  "
                  f"{entry['tokens_per_sec']:10.1f} tok/s  "
                  f"peak {entry['peak_mem_gb']:.2f} GB")
            torch.cuda.empty_cache()

    save(rows, args.out, {
        "model": "cfm", "checkpoint": args.checkpoint, "checkpoint_step": step,
        "sampler": kind, "arch": {k: v for k, v in arch.items() if not k.startswith("_")},
        "n_params": n_params, "gpu": gpu_name(), "device": args.device,
        "torch": torch.__version__, "warmup": args.warmup, "repeats": args.repeats,
        "dtype": args.amp or "fp32", "fast_sampler": args.fast,
        "compiled": args.compile,
        "tf32": False if args.amp is None else None,
    })


if __name__ == "__main__":
    main()
