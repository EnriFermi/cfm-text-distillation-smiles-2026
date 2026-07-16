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
    flop_cost_per_token_full_recompute,
    flop_cost_per_token_masked,
    forward_token_cost,
    nfe_per_token,
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_bcfm.yaml")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    device = _resolve_device(cfg.device)
    length = cfg.data.k
    sampler = cfg.sampler
    is_gold = sampler.name == "gold"
    block_size = sampler.get("block_size", length)  # full_cfm == one block of length L
    report_block_size = cfg.get("report_block_size") or block_size
    # number of flow-map jumps per block; an explicit schedule overrides steps_per_block
    n_jumps = (len(sampler.schedule) if sampler.get("schedule")
               else sampler.get("steps_per_block", sampler.get("steps", 1)))
    if is_gold:
        n_jumps = 0

    need_dm = cfg.detok or is_gold or sampler.get("prefix") == "gold"
    dm = make_datamodule(cfg, device) if need_dm else None

    sampling_seconds = None
    if is_gold:
        tokens = gold_sequences(dm, cfg.n_samples, device).cpu()
    else:
        module = load_module(cfg, device)
        gold_prefix = (gold_sequences(dm, cfg.n_samples, device)
                       if sampler.get("prefix") == "gold" else None)
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

    if sampler.name == "bcfm_infer":
        flop_cost = flop_cost_per_token_masked(block_size, n_jumps, length)
        cost_model = "masked_2L"
    elif sampler.name == "bcfm_train":
        flop_cost = forward_token_cost(block_size, n_jumps, length)
        cost_model = "cached"
    else:
        flop_cost = flop_cost_per_token_full_recompute(block_size, n_jumps, length)
        cost_model = "full_recompute"

    ent = entropy_summary(tokens, report_block_size)
    metrics = {
        "model": cfg.get("model_id") or ("data" if is_gold else sampler.name),
        "sampler": sampler.name,
        "length": int(length),
        "block_size": int(block_size),
        "steps_per_block": int(n_jumps),
        "nfe_per_token": nfe_per_token(block_size, n_jumps, length),
        "flop_cost_per_token": flop_cost,
        "cost_model": cost_model,
        "seed": int(cfg.seed),
        "prefix": sampler.get("prefix", "generated"),
        "discretize": sampler.get("discretize", "argmax"),
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

    out_dir = Path(cfg.paths.root_dir) / "results" / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if strings is not None:
        (out_dir / "samples.txt").write_text("\n".join(strings[:64]))
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(json.dumps({k: metrics[k] for k in (
        "model", "sampler", "length", "block_size", "steps_per_block",
        "nfe_per_token", "flop_cost_per_token", "gen_ppl",
        "mean_entropy_pooled", "mean_entropy_per_sample",
        "entropy_block0_pooled", "entropy_block_last_pooled",
        "tokens_per_sec",
    )}, indent=2))


if __name__ == "__main__":
    main()
