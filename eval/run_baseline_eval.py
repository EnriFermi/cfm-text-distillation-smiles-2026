"""Evaluate colleague AR / MDLM / BD3-LM checkpoints with the same gen-PPL harness as M1–M3.

Loads ``bcfm_baselines`` from ``BASELINES_CODE`` (default ``<repo>/baselines``), generates
Text8 strings, scores GPT-J gen-PPL, and writes ``results/<exp>/metrics.json`` in
``network_forwards_v2`` so ``python -m eval.aggregate`` picks them up next to M1/M2/M3.

With ``--latency-only``, updates an existing metrics.json with batch-size-one end-to-end
latency (same protocol as ``eval.run_eval`` / ``scripts/measure_sequence_latency.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import rootutils
import torch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from eval.metrics import (  # noqa: E402
    entropy_per_block,
    entropy_per_block_per_sample,
    entropy_summary,
    gen_ppl,
)


def _baselines_code() -> Path:
    env = os.environ.get("BASELINES_CODE")
    if env:
        return Path(env)
    return ROOT / "baselines"


def _ensure_baselines_on_path() -> Path:
    code = _baselines_code()
    if not (code / "bcfm_baselines").is_dir():
        raise FileNotFoundError(f"bcfm_baselines not found under {code}")
    sys.path.insert(0, str(code))
    return code


def _default_text8() -> Path:
    return Path(os.environ.get(
        "TEXT8_DATA_DIR",
        str(ROOT / "data" / "text8"),
    ))


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_sampling(args, config: dict) -> dict:
    sampling = dict(config["sampling"])
    sampling.update({
        "length": args.length,
        "batch_size": min(args.batch_size, args.n_samples),
        "seed": args.seed,
    })
    if args.num_steps is not None:
        sampling["num_steps"] = args.num_steps
    if args.steps_per_block is not None:
        sampling["steps_per_block"] = args.steps_per_block
    if args.first_hitting is not None:
        sampling["first_hitting"] = args.first_hitting
    return sampling


def _measure_sequence_latency_ms(
    model,
    sampling: dict,
    *,
    device: str,
    warmup_runs: int,
    repeats: int,
) -> dict[str, float | int]:
    """End-to-end latency for one sequence (batch_size=1), synced CUDA brackets."""
    from bcfm_baselines.models.generative_text import GenerationConfig

    if repeats < 1:
        raise ValueError("latency_repeats must be at least 1")

    gen_cfg = GenerationConfig(**{**sampling, "batch_size": 1})
    for _ in range(warmup_runs):
        model.generate(gen_cfg)
    _sync(device)

    samples_ms: list[float] = []
    for i in range(repeats):
        gen_cfg.seed = int(sampling.get("seed", 0)) + i
        _sync(device)
        start = time.perf_counter()
        model.generate(gen_cfg)
        _sync(device)
        samples_ms.append((time.perf_counter() - start) * 1_000)

    samples_ms.sort()

    def percentile(q: float) -> float:
        return samples_ms[round((len(samples_ms) - 1) * q)]

    return {
        "sequence_latency_ms": st.median(samples_ms),
        "sequence_latency_p10_ms": percentile(0.10),
        "sequence_latency_p90_ms": percentile(0.90),
        "sequence_latency_repeats": repeats,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=None, help="MDLM diffusion steps")
    p.add_argument("--steps-per-block", type=int, default=None, help="BD3-LM steps/block")
    p.add_argument("--first-hitting", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--judge-model", default="EleutherAI/gpt-j-6b")
    p.add_argument("--judge-batch-size", type=int, default=4)
    p.add_argument("--report-block-size", type=int, default=16)
    p.add_argument("--text8-data-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument(
        "--latency-only",
        action="store_true",
        help="update existing metrics.json with batch=1 latency; skip sampling/judge",
    )
    p.add_argument("--latency-warmup-runs", type=int, default=5)
    p.add_argument("--latency-repeats", type=int, default=30)
    args = p.parse_args()

    _ensure_baselines_on_path()
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.data import create_corpus
    from bcfm_baselines.models.generative_text import GenerationConfig

    text8 = args.text8_data_dir or _default_text8()
    out_dir = args.results_dir / args.exp_name
    metrics_file = out_dir / "metrics.json"

    model, payload = load_checkpoint_model(args.checkpoint, args.device)
    config = payload["config"]
    config["dataset"]["data_dir"] = str(text8)
    sampling = _build_sampling(args, config)

    if args.latency_only:
        if not metrics_file.exists():
            raise FileNotFoundError(
                f"latency_only requires existing metrics at {metrics_file}"
            )
        # Prefer step settings already recorded in metrics if CLI omitted them.
        existing = json.loads(metrics_file.read_text())
        if args.num_steps is None and existing.get("sampler") == "mdlm":
            sampling["num_steps"] = int(existing["steps_per_block"])
        if args.steps_per_block is None and existing.get("sampler") == "bd3lm":
            sampling["steps_per_block"] = int(existing["steps_per_block"])
        if args.first_hitting is None and "first_hitting" in existing:
            sampling["first_hitting"] = existing["first_hitting"]

        latency = _measure_sequence_latency_ms(
            model,
            sampling,
            device=args.device,
            warmup_runs=args.latency_warmup_runs,
            repeats=args.latency_repeats,
        )
        existing.update(latency)
        existing["latency_device"] = args.device
        metrics_file.write_text(json.dumps(existing, indent=2))
        print(f"Updated {metrics_file} with {latency}")
        return

    corpus = create_corpus(config)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokens_chunks = []
    strings: list[str] = []
    nfe_total = 0
    t0 = time.perf_counter()
    remaining = args.n_samples
    while remaining > 0:
        bs = min(sampling["batch_size"], remaining)
        gen_cfg = GenerationConfig(**{**sampling, "batch_size": bs})
        # Keep RNG reproducible across chunks: advance seed per chunk.
        gen_cfg.seed = args.seed + (args.n_samples - remaining)
        result = model.generate(gen_cfg)
        tokens_chunks.append(result.tokens.cpu())
        strings.extend(corpus.decode(result.tokens))
        nfe_total = int(result.diagnostics.get("num_function_evaluations", nfe_total))
        remaining -= bs
    _sync(args.device)
    sampling_seconds = time.perf_counter() - t0
    tokens = torch.cat(tokens_chunks, dim=0)
    nfe_per_seq = nfe_total

    mtype = model.model_type  # ar | mdlm | bd3lm
    model_id = {"ar": "AR", "mdlm": "MDLM", "bd3lm": "BD3LM"}[mtype]
    if mtype == "bd3lm":
        block_size = int(config["model"].get("block_size", args.report_block_size))
        steps = int(sampling.get("steps_per_block") or sampling.get("num_steps") or 1)
    elif mtype == "mdlm":
        block_size = int(args.length)
        steps = int(sampling.get("num_steps") or 64)
    else:
        block_size = 1
        steps = 1

    ent = entropy_summary(tokens, args.report_block_size)
    metrics = {
        "metric_schema": "network_forwards_v2",
        "model": model_id,
        "sampler": mtype,
        "length": int(args.length),
        "block_size": block_size,
        "steps_per_block": steps,
        "flow_nfe_total": nfe_per_seq,
        "flow_nfe_per_token": nfe_per_seq / args.length,
        "cache_encode_forwards": 0,
        "nfe_total": nfe_per_seq,
        "nfe_per_token": nfe_per_seq / args.length,
        "context_token_cost_per_token": None,
        "cost_model": "baseline_diagnostics_nfe",
        "seed": int(args.seed),
        "prefix": "generated",
        "discretize": sampling.get("strategy", "sample"),
        "use_kv_cache": bool(sampling.get("use_kv_cache", True)),
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
        "first_hitting": sampling.get("first_hitting"),
        "weights_used_for_inference": payload.get("weights_used_for_inference"),
        "checkpoint_step": int(payload.get("step", -1)),
        "sampling_config": {k: v for k, v in sampling.items() if k != "seed"},
    }

    if args.judge:
        print(f"Computing generative perplexity ({args.judge_model}) ...")
        metrics["gen_ppl"] = gen_ppl(
            strings,
            batch_size=args.judge_batch_size,
            context_size=args.length,
            model=args.judge_model,
            device=args.device,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))
    (out_dir / "samples.txt").write_text("\n".join(strings[:64]))
    print(f"Wrote {metrics_file}")
    print(json.dumps({
        k: metrics[k] for k in (
            "model", "sampler", "block_size", "steps_per_block", "nfe_total",
            "gen_ppl", "tokens_per_sec", "first_hitting", "checkpoint_step",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
