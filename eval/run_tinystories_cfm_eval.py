"""TinyStories CFM eval for NEW.zip M1 checkpoint (GPT-2 vocab DiT) + M2 block wrap.

Writes ``results/<exp>/metrics.json`` in ``network_forwards_v2`` (same schema as Text8).
Detok uses HuggingFace ``gpt2`` (training tokenizer), judge is GPT-J gen-PPL.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from block.nfe import (  # noqa: E402
    cache_encode_forwards,
    context_token_cost_per_token,
    flow_forwards,
    full_sequence_token_cost,
    network_forwards,
    nfe_per_token,
)
from eval.generate import load_module, sample_tokens  # noqa: E402
from eval.metrics import (  # noqa: E402
    entropy_per_block,
    entropy_per_block_per_sample,
    entropy_summary,
    gen_ppl,
)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _model_cfg(length: int = 256) -> OmegaConf:
    return OmegaConf.create({
        "_target_": "semicat.models.textsemicat.TextSemicatModule",
        "in_shape": [length, 50257],
        "prior_type": "gaussian",
        "scheduler": None,
        "sd_type": "lag",
        "sd_prop": 0.25,
        "optimizer": None,
        "compile": False,
        "calc_nll": False,
        "net": {
            "_target_": "semicat.net.duo.DIT",
            "vocab_size": 50257,
            "hidden_size": 384,
            "cond_dim": 128,
            "n_blocks": 6,
            "n_heads": 6,
            "dropout": 0.1,
            "length": length,
            "embed_type": "rms",
        },
    })


def _decode_gpt2(tokens: torch.Tensor) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    return [tok.decode(row.tolist(), skip_special_tokens=True) for row in tokens]


def _measure_latency_ms(module, sampler, length: int, device: str,
                        warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        sample_tokens(module, sampler, n_samples=1, batch_size=1, length=length)
    _sync(device)
    samples_ms = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        sample_tokens(module, sampler, n_samples=1, batch_size=1, length=length)
        _sync(device)
        samples_ms.append((time.perf_counter() - t0) * 1000)
    samples_ms.sort()

    def pct(q: float) -> float:
        return samples_ms[round((len(samples_ms) - 1) * q)]

    return {
        "sequence_latency_ms": st.median(samples_ms),
        "sequence_latency_p10_ms": pct(0.10),
        "sequence_latency_p90_ms": pct(0.90),
        "sequence_latency_repeats": repeats,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--sampler", choices=["full_cfm", "bcfm_infer"], default="full_cfm")
    p.add_argument("--steps", type=int, default=1, help="M1 sampling steps")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps-per-block", type=int, default=1)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--judge-model", default="EleutherAI/gpt-j-6b")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--report-block-size", type=int, default=16)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--latency-only", action="store_true")
    p.add_argument("--latency-warmup-runs", type=int, default=5)
    p.add_argument("--latency-repeats", type=int, default=30)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = args.results_dir / args.exp_name
    metrics_file = out_dir / "metrics.json"

    if args.sampler == "full_cfm":
        sampler = OmegaConf.create({
            "name": "full_cfm", "steps": args.steps, "discretize": "argmax",
        })
        block_size = args.length
        n_jumps = args.steps
        model_id = args.model_id or "M1-TS"
        uses_kv = False
    else:
        sampler = OmegaConf.create({
            "name": "bcfm_infer",
            "block_size": args.block_size,
            "steps_per_block": args.steps_per_block,
            "discretize": "argmax",
            "prefix": "generated",
            "schedule": None,
            "use_kv_cache": True,
        })
        block_size = args.block_size
        n_jumps = args.steps_per_block
        model_id = args.model_id or "M2-TS"
        uses_kv = True

    cfg = OmegaConf.create({
        "ckpt_path": str(args.checkpoint),
        "model": _model_cfg(args.length),
        "seed": args.seed,
    })
    module = load_module(cfg, args.device)

    if args.latency_only:
        if not metrics_file.exists():
            raise FileNotFoundError(f"missing {metrics_file}")
        latency = _measure_latency_ms(
            module, sampler, args.length, args.device,
            args.latency_warmup_runs, args.latency_repeats,
        )
        metrics = json.loads(metrics_file.read_text())
        metrics.update(latency)
        metrics["latency_device"] = args.device
        metrics_file.write_text(json.dumps(metrics, indent=2))
        print(f"Updated {metrics_file} with {latency}")
        return

    _sync(args.device)
    t0 = time.perf_counter()
    tokens = sample_tokens(
        module, sampler,
        n_samples=args.n_samples, batch_size=args.batch_size, length=args.length,
    )
    _sync(args.device)
    sampling_seconds = time.perf_counter() - t0

    flow_nfe = flow_forwards(block_size, n_jumps, args.length)
    cache_builds = cache_encode_forwards(block_size, args.length) if uses_kv else 0
    nfe_total = network_forwards(block_size, n_jumps, args.length, uses_kv_cache=uses_kv)
    if uses_kv:
        context_cost = context_token_cost_per_token(block_size, n_jumps, args.length)
        cost_model = "cached_context_token_proxy"
    else:
        context_cost = full_sequence_token_cost(block_size, n_jumps, args.length)
        cost_model = "full_sequence_token_proxy"

    ent = entropy_summary(tokens, args.report_block_size)
    metrics = {
        "metric_schema": "network_forwards_v2",
        "model": model_id,
        "sampler": args.sampler,
        "dataset": "tinystories",
        "length": int(args.length),
        "block_size": int(block_size),
        "steps_per_block": int(n_jumps),
        "flow_nfe_total": int(flow_nfe),
        "flow_nfe_per_token": flow_nfe / args.length,
        "cache_encode_forwards": int(cache_builds),
        "nfe_total": int(nfe_total),
        "nfe_per_token": nfe_per_token(
            block_size, n_jumps, args.length, uses_kv_cache=uses_kv,
        ),
        "context_token_cost_per_token": context_cost,
        "cost_model": cost_model,
        "seed": int(args.seed),
        "prefix": "generated",
        "discretize": "argmax",
        "use_kv_cache": uses_kv if uses_kv else None,
        "schedule": None,
        "report_block_size": int(args.report_block_size),
        "entropy_per_block": entropy_per_block(tokens, args.report_block_size),
        "entropy_per_block_ps": entropy_per_block_per_sample(tokens, args.report_block_size),
        **ent,
        "gen_ppl": None,
        "sampling_seconds": sampling_seconds,
        "tokens_per_sec": (args.n_samples * args.length / sampling_seconds
                           if sampling_seconds else None),
        "ckpt": str(args.checkpoint.resolve()),
        "n_samples": int(args.n_samples),
        "tokenizer": "gpt2",
    }

    strings = _decode_gpt2(tokens)
    if args.judge:
        print(f"Computing generative perplexity ({args.judge_model}) ...")
        metrics["gen_ppl"] = gen_ppl(
            strings, batch_size=args.judge_batch_size,
            context_size=args.length, model=args.judge_model, device=args.device,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))
    (out_dir / "samples.txt").write_text("\n".join(strings[:64]))
    print(f"Wrote {metrics_file}")
    print(json.dumps({
        k: metrics[k] for k in (
            "model", "sampler", "block_size", "steps_per_block", "nfe_total",
            "gen_ppl", "tokens_per_sec",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
