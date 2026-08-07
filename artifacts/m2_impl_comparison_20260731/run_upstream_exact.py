"""Run the exact M2 generator from a checked-out upstream worktree.

This file is orchestration only.  The model builder and sampler are imported from
``--source-root`` so the implementation under test remains byte-for-byte the code
at that Git revision.  The output follows ``bcfm-textdump/v1`` and preserves token
IDs for scorer-independent comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    out = args.out.resolve()
    sys.path.insert(0, str(source_root))

    import torch
    from omegaconf import OmegaConf

    import eval.generate as generate_module
    from eval.generate import load_module, sample_tokens
    from eval.run_tinystories_cfm_eval import _decode_gpt2, _model_cfg

    imported_sources = {
        "sample_tokens": Path(inspect.getfile(inspect.unwrap(sample_tokens))).resolve(),
        "blockwise_sample": Path(
            inspect.getfile(inspect.unwrap(generate_module.blockwise_sample))
        ).resolve(),
    }
    for symbol, imported_path in imported_sources.items():
        if source_root not in imported_path.parents:
            raise RuntimeError(
                f"{symbol} came from {imported_path}, not {source_root}"
            )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source_root, text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"upstream worktree is dirty:\n{dirty}")

    sampler = OmegaConf.create(
        {
            "name": "bcfm_infer",
            "block_size": args.block_size,
            "steps_per_block": args.nfe,
            "discretize": "argmax",
            "prefix": "generated",
            "schedule": None,
            "use_kv_cache": True,
        }
    )
    cfg = OmegaConf.create(
        {
            "ckpt_path": str(checkpoint),
            "model": _model_cfg(args.length),
            "seed": args.seed,
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    resolved = {
        "implementation": "upstream_clean_prefix_kv",
        "source_root": str(source_root),
        "source_commit": commit,
        "source_dirty": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "device": args.device,
        "seed": args.seed,
        "length": args.length,
        "block_size": args.block_size,
        "nfe": args.nfe,
        "schedule": "uniform",
        "discretize": "argmax",
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "output": str(out),
        "source_files": {
            rel: sha256(source_root / rel)
            for rel in (
                "block/sampling.py",
                "semicat/net/duo.py",
                "eval/generate.py",
                "eval/run_tinystories_cfm_eval.py",
            )
        },
        "imported_sources": {key: str(value) for key, value in imported_sources.items()},
    }
    print("[config] " + json.dumps(resolved, indent=2), flush=True)
    print("[stage] model build + strict checkpoint load", flush=True)
    module = load_module(cfg, args.device)
    first_parameter = next(module.parameters())
    resolved["dtype"] = str(first_parameter.dtype)
    resolved["module_class"] = type(module).__name__
    resolved["net_class"] = type(module.net).__name__
    imported_net = Path(inspect.getfile(type(module.net))).resolve()
    if source_root not in imported_net.parents:
        raise RuntimeError(f"model class came from {imported_net}, not {source_root}")
    resolved["imported_sources"]["net_class"] = str(imported_net)
    print(
        f"[model] module={resolved['module_class']} net={resolved['net_class']} "
        f"dtype={resolved['dtype']}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(
        f"[stage] sampling label=upstream seed={args.seed} B={args.block_size} "
        f"NFE={args.nfe} batches={(args.n_samples + args.batch_size - 1) // args.batch_size}",
        flush=True,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    tokens = sample_tokens(
        module,
        sampler,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        length=args.length,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    texts = _decode_gpt2(tokens)

    point_id = f"argmax_nfe{args.nfe}"
    n_blocks = args.length // args.block_size
    flow_forwards = n_blocks * args.nfe
    cache_forwards = n_blocks - 1
    payload = {
        "schema": "bcfm-textdump/v1",
        "label": "upstream_m2_clean_prefix_kv",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit[:7],
        "implementation": resolved,
        "model": {
            "kind": "checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": resolved["checkpoint_sha256"],
            "dataset": "tinystories",
            "tokenizer": "gpt2",
            "arch": {
                "vocab_size": 50257,
                "hidden_size": 384,
                "length": args.length,
                "n_heads": 6,
                "n_blocks": 6,
                "cond_dim": 128,
                "embed_type": "rms",
            },
        },
        "sampler": {
            "kind": "bcfm_infer",
            "algorithm": "clean_prefix_kv_cache",
            "block_size": args.block_size,
            "length": args.length,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
        "points": [
            {
                "id": point_id,
                "nfe": args.nfe,
                "discretize": "argmax",
                "forwards_per_sequence": flow_forwards + cache_forwards,
                "flow_forwards_per_sequence": flow_forwards,
                "cache_encode_forwards": cache_forwards,
                "n_samples": int(tokens.shape[0]),
                "sampling_seconds": round(elapsed, 6),
                "texts": texts,
            }
        ],
    }
    out.write_text(json.dumps(payload, indent=1))
    np.savez_compressed(
        out.with_suffix(".tokens.npz"), **{point_id: tokens.numpy().astype(np.int32)}
    )
    out.with_name(out.stem + ".manifest.json").write_text(
        json.dumps(resolved, indent=2) + "\n"
    )
    print(
        f"[done] elapsed={elapsed:.3f}s tokens={tuple(tokens.shape)} "
        f"artifacts={out},{out.with_suffix('.tokens.npz')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
