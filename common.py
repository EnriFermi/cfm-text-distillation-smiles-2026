"""Shared setup for the latency benchmarks.

Both models live in sibling repositories that are not installed as packages, so
every entry point calls :func:`bootstrap` first: it puts the right repo on
``sys.path`` and, for the CFM side, the pure-torch ``flash_attn`` shim in
``shim/`` (see its docstring — ``semicat/net/duo.py`` imports flash-attn at
module level but only uses one pure-torch function from it).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CFM_REPO = PROJECT / "cfm-text-distillation-smiles-2026"
BD3_REPO = PROJECT / "BCFM Baselines"
CFM_CKPT = PROJECT / "2 models" / "cfm_mod" / "last.ckpt"
BD3_CKPT = PROJECT / "2 models" / "bd3lm" / "last.pt"
MDLM_CKPT = (
    PROJECT
    / "2 models"
    / "mdlm_tinystories_gpt2_h384_l6_seed12345"
    / "checkpoints"
    / "last.pt"
)
RESULTS = HERE / "results"


def bootstrap(which: str) -> None:
    """``which`` is ``"cfm``, ``"bd3lm``, or ``"mdlm"``."""
    if which == "cfm":
        sys.path.insert(0, str(HERE / "shim"))
        sys.path.insert(0, str(CFM_REPO))
    elif which in {"bd3lm", "mdlm"}:
        sys.path.insert(0, str(BD3_REPO))
    else:
        raise ValueError(which)


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def timed(fn, *, warmup: int = 2, repeats: int = 5) -> dict:
    """Wall-clock a GPU callable properly: warm up, then sync-bracket every run.

    Returns median/mean/min/max in seconds. The median is what we report — a
    single run on a desktop GPU picks up clock and compositor noise.
    """
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "runs": samples,
    }


def row(*, model: str, batch_size: int, length: int, stats: dict, **extra) -> dict:
    """One benchmark point, in the shape every table downstream reads."""
    median = stats["median_s"]
    out = {
        "model": model,
        "batch_size": batch_size,
        "length": length,
        "latency_s_per_batch": round(median, 4),
        "latency_ms_per_sequence": round(1000 * median / batch_size, 2),
        "tokens_per_sec": round(batch_size * length / median, 1),
        "min_s": round(stats["min_s"], 4),
        "max_s": round(stats["max_s"], 4),
    }
    out.update(extra)
    return out


def save(rows: list[dict], name: str, meta: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"\n[saved] {path}")
    return path


def peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9
