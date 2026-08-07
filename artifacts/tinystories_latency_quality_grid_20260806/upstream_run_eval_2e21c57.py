"""Hydra entry point: checkpoint + sampler -> results/<exp>/metrics.json (+ samples.txt).

Evaluates ONE sampler setting. Sweep with Hydra multirun, e.g.::

    python -m eval.run_eval -m ckpt_path=/path/to/m1.ckpt sampler=bcfm_infer \\
        sampler.block_size=4,8,16 sampler.steps_per_block=1,2,4

Each job writes its own results file, so parallel sweeps never collide. The results
schema is documented in CLAUDE.md; downstream plots/tables (eval/aggregate.py) read it.

``sampler=gold`` scores real test data instead of a model — the reference lines for
gen-PPL and entropy that every figure needs. Custom-``schedule`` runs must set
``exp_name=...`` manually (the schedule is not encoded in the default name).
"""

from __future__ import annotations

import json
import statistics as st
import subprocess
import time
from pathlib import Path

import hydra
import rootutils
import torch
from lightning import seed_everything
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from block.nfe import (  # noqa: E402
    cache_encode_forwards,
    context_token_cost_per_token,
    flow_forwards,
    full_sequence_token_cost,
    masked_2l_token_cost,
    nfe_per_token,
    network_forwards,
)
from eval.generate import load_module, make_datamodule, sample_tokens, gold_sequences  # noqa: E402
from eval.metrics import (  # noqa: E402
    entropy_per_block,
    entropy_per_block_per_sample,
    entropy_summary,
    gen_ppl,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def _sync(device: str) -> None:
    """Block until queued device work finishes, so wall-clock timing is honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _measure_sequence_latency_ms(
    module,
    sampler: DictConfig,
    *,
    length: int,
    device: str,
    warmup_runs: int,
    repeats: int,
) -> dict[str, float | int]:
    """Measure end-to-end latency for one generated sequence.

    Each timing interval is bracketed by device synchronization, so asynchronous CUDA
    execution cannot leak into the next sample or escape the measured interval. Warmup
    uses the same batch-size-one path and is excluded from the reported distribution.
    """
    if repeats < 1:
        raise ValueError("latency_repeats must be at least 1")

    for _ in range(warmup_runs):
        sample_tokens(module, sampler, n_samples=1, batch_size=1, length=length)
    _sync(device)

    samples_ms = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        sample_tokens(module, sampler, n_samples=1, batch_size=1, length=length)
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_bcfm.yaml")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    device = _resolve_device(cfg.device)
    sampler = cfg.sampler
    is_gold = sampler.name == "gold"
    block_size = sampler.get("block_size", cfg.data.k)  # full_cfm == one block of length L
    if sampler.get("num_blocks") is not None:
        length = int(sampler.num_blocks) * int(block_size)
    else:
        length = int(cfg.get("length") or cfg.data.k)
    report_block_size = cfg.get("report_block_size") or block_size
    # number of flow-map jumps per block; an explicit schedule overrides steps_per_block
    n_jumps = (len(sampler.schedule) if sampler.get("schedule")
               else sampler.get("steps_per_block", sampler.get("steps", 1)))
    if is_gold:
        n_jumps = 0
    out_dir = Path(cfg.paths.root_dir) / "results" / cfg.exp_name

    if cfg.get("latency_only", False):
        metrics_file = out_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(
                f"latency_only requires existing metrics at {metrics_file}"
            )
        if is_gold:
            raise ValueError("latency_only is not defined for sampler=gold")

        module = load_module(cfg, device)
        latency = _measure_sequence_latency_ms(
            module,
            sampler,
            length=length,
            device=device,
            warmup_runs=int(cfg.get("latency_warmup_runs", 5)),
            repeats=int(cfg.get("latency_repeats", 30)),
        )
        metrics = json.loads(metrics_file.read_text())
        metrics.update(latency)
        metrics["latency_device"] = device
        metrics_file.write_text(json.dumps(metrics, indent=2))
        print(f"Updated {metrics_file} with {latency}")
        return

    need_dm = cfg.detok or is_gold or sampler.get("prefix") == "gold"
    dm = make_datamodule(cfg, device) if need_dm else None

    sampling_seconds = None
    if is_gold:
        tokens = gold_sequences(dm, cfg.n_samples, device).cpu()
    else:
        module = load_module(cfg, device)
        gold_prefix = (gold_sequences(dm, cfg.n_samples, device)
                       if sampler.get("prefix") == "gold" else None)
        if gold_prefix is not None and gold_prefix.shape[-1] != length:
            gold_prefix = gold_prefix[:, :length]
        # Warm up before timing: the first forward pays cudnn autotune + torch.compile,
        # which would make tokens_per_sec (E13) unrepresentative. One throwaway batch,
        # then a hard sync, so t0 starts from steady state.
        warm_n = min(cfg.batch_size, 8)
        sample_tokens(module, sampler, n_samples=warm_n, batch_size=warm_n, length=length,
                      gold_prefix=(gold_prefix[:warm_n] if gold_prefix is not None else None))
        _sync(device)
        seed_everything(cfg.seed, workers=True)  # warmup consumed RNG; make samples warmup-invariant
        print(f"Sampling {cfg.n_samples} sequences with sampler={sampler.name} ...")
        t0 = time.perf_counter()
        tokens = sample_tokens(
            module, sampler,
            n_samples=cfg.n_samples, batch_size=cfg.batch_size,
            length=length, gold_prefix=gold_prefix,
        )
        _sync(device)
        sampling_seconds = time.perf_counter() - t0

    uses_kv_cache = (
        sampler.name == "bcfm_infer"
        or (sampler.name == "bcfm_train" and bool(sampler.get("use_kv_cache", True)))
    )
    flow_nfe_total = flow_forwards(block_size, n_jumps, length)
    cache_builds = cache_encode_forwards(block_size, length) if uses_kv_cache else 0
    nfe_total = network_forwards(
        block_size, n_jumps, length, uses_kv_cache=uses_kv_cache,
    )

    if uses_kv_cache:
        context_cost = context_token_cost_per_token(block_size, n_jumps, length)
        cost_model = "cached_context_token_proxy"
    elif sampler.name == "bcfm_train":
        context_cost = masked_2l_token_cost(block_size, n_jumps, length)
        cost_model = "masked_2L_token_proxy"
    else:
        context_cost = full_sequence_token_cost(block_size, n_jumps, length)
        cost_model = "full_sequence_token_proxy"

    ent = entropy_summary(tokens, report_block_size)
    metrics = {
        "metric_schema": "network_forwards_v2",
        "model": cfg.get("model_id") or ("data" if is_gold else sampler.name),
        "sampler": sampler.name,
        "length": int(length),
        "block_size": int(block_size),
        "steps_per_block": int(n_jumps),
        "flow_nfe_total": int(flow_nfe_total),
        "flow_nfe_per_token": flow_nfe_total / length,
        "cache_encode_forwards": int(cache_builds),
        "nfe_total": int(nfe_total),
        "nfe_per_token": nfe_per_token(
            block_size, n_jumps, length, uses_kv_cache=uses_kv_cache,
        ),
        "context_token_cost_per_token": context_cost,
        "cost_model": cost_model,
        "seed": int(cfg.seed),
        "prefix": sampler.get("prefix", "generated"),
        "discretize": sampler.get("discretize", "argmax"),
        "use_kv_cache": (uses_kv_cache if sampler.name in ("bcfm_train", "bcfm_infer") else None),
        "schedule": ([list(st) for st in sampler.schedule] if sampler.get("schedule") else None),
        "report_block_size": int(report_block_size),
        "entropy_per_block": entropy_per_block(tokens, report_block_size),
        "entropy_per_block_ps": entropy_per_block_per_sample(tokens, report_block_size),
        **ent,
        "gen_ppl": None,
        "sampling_seconds": sampling_seconds,
        "tokens_per_sec": (cfg.n_samples * length / sampling_seconds if sampling_seconds else None),
        "ckpt": cfg.get("ckpt_path"),
        "commit": _git_commit(),
        "n_samples": int(cfg.n_samples),
    }

    strings = dm.tensor_to_strings(tokens) if cfg.detok else None
    if cfg.judge:
        assert strings is not None, "judge=true requires detok=true"
        print("Computing generative perplexity ...")
        metrics["gen_ppl"] = gen_ppl(
            strings, batch_size=cfg.judge_batch_size,
            context_size=length, model=cfg.judge_model, device=device,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if strings is not None:
        (out_dir / "samples.txt").write_text("\n".join(strings[:64]))
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(json.dumps({k: metrics[k] for k in (
        "model", "sampler", "length", "block_size", "steps_per_block",
        "flow_nfe_total", "cache_encode_forwards", "nfe_total", "nfe_per_token",
        "context_token_cost_per_token", "gen_ppl",
        "mean_entropy_pooled", "mean_entropy_per_sample",
        "entropy_block0_pooled", "entropy_block_last_pooled",
        "tokens_per_sec",
    )}, indent=2))


if __name__ == "__main__":
    main()
