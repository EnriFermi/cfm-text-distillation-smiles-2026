"""Measure batch=1 latency for the exact TinyStories quality-evaluation grid.

The timing functions are loaded directly (via their AST nodes) from
``eval/run_eval.py`` in an upstream git worktree.  Their implementation is therefore
executed unchanged.  Model construction and sampling come from ``eval.dump_samples``
in the current tree so every timed point uses the same checkpoint, architecture,
sampler, NFE and discretization as the stored MAUVE/gen-PPL dump.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import statistics as st
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from eval.dump_samples import build_module, sample_point


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "tinystories_latency_quality_grid_20260806"
EXPECTED_UPSTREAM_COMMIT = "2e21c57704ab94e131e3350ffd8286632f61a7b8"

MODELS = (
    {
        "name": "M1-100k-source",
        "dump": ROOT / "dumps" / "ts_source_M1_full.json",
        "scores": ROOT / "results" / "ts_M1_M2_M3_scores.json",
    },
    {
        "name": "M2-source-blockcausal",
        "dump": ROOT / "dumps" / "ts_M2_source.json",
        "scores": ROOT / "results" / "ts_M1_M2_M3_scores.json",
    },
    {
        "name": "M2-source-fast",
        "dump": ROOT / "dumps" / "ts_M2_source_fast_cached_argmax.json",
        "scores": ROOT / "results" / "ts_M2_source_fast_cached_argmax_scores.json",
    },
    {
        "name": "M3-step100000",
        "dump": ROOT / "dumps" / "ts_M3_step100000.json",
        "scores": ROOT / "results" / "ts_M1_M2_M3_scores.json",
    },
    {
        "name": "M3-step100000-fast",
        "dump": ROOT / "dumps" / "ts_M3_step100000_fast_cached.json",
        "scores": ROOT / "results" / "ts_M3_step100000_fast_cached_scores.json",
    },
    {
        "name": "M1-step200000",
        "dump": ROOT / "dumps" / "ts_M1_step200000_full.json",
        "scores": ROOT / "results" / "ts_M1_step200000_full_scores.json",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_upstream_timer(source_path: Path, sample_tokens_fn):
    """Compile the two timing functions directly from the upstream source AST."""
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    wanted = {"_sync", "_measure_sequence_latency_ms"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    found = {node.name for node in nodes}
    if found != wanted:
        raise RuntimeError(f"missing upstream timing functions: {sorted(wanted - found)}")
    extracted = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace = {
        "torch": torch,
        "st": st,
        "time": time,
        "DictConfig": DictConfig,
        "sample_tokens": sample_tokens_fn,
    }
    exec(compile(extracted, str(source_path), "exec"), namespace)
    return namespace["_measure_sequence_latency_ms"]


def _write_outputs(output_dir: Path, metadata: dict, rows: list[dict]) -> None:
    payload = {"metadata": metadata, "rows": rows}
    json_path = output_dir / "latency_quality_grid.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not rows:
        return
    csv_path = output_dir / "latency_quality_grid.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], output_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    colors = {
        "M1-100k-source": "#1f77b4",
        "M2-source-blockcausal": "#ff7f0e",
        "M2-source-fast": "#e377c2",
        "M3-step100000": "#2ca02c",
        "M3-step100000-fast": "#9467bd",
        "M1-step200000": "#d62728",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, mode in zip(axes, ("argmax", "sample")):
        for model in colors:
            points = sorted(
                (row for row in rows if row["model"] == model and row["discretize"] == mode),
                key=lambda row: row["nfe"],
            )
            if not points:
                continue
            xs = [row["sequence_latency_ms"] for row in points]
            ys = [row[metric] for row in points]
            xerr = [
                [x - row["sequence_latency_p10_ms"] for row, x in zip(points, xs)],
                [row["sequence_latency_p90_ms"] - x for row, x in zip(points, xs)],
            ]
            ax.errorbar(xs, ys, xerr=xerr, marker="o", linewidth=1.8,
                        elinewidth=0.8, capsize=2, alpha=0.95,
                        color=colors[model], label=model)
            for row, x, y in zip(points, xs, ys):
                if row["nfe"] >= 4:
                    ax.annotate(str(row["nfe"]), (x, y), xytext=(3, 3),
                                textcoords="offset points", fontsize=7)
        ax.set_xscale("log")
        ax.set_title(mode)
        ax.set_xlabel("batch=1 sequence latency, median ms (log scale)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"TinyStories: {ylabel} vs batch=1 latency (point labels are NFE)",
        y=0.99,
    )
    # Reserve a stable title band.  Without it, bbox_inches='tight' can place the
    # subplot titles on top of the suptitle for the taller gen-PPL y-range.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-worktree", type=Path,
        default=Path("/tmp/cfm-upstream-m2-latency-2e21c57"),
    )
    parser.add_argument("--upstream-commit", default=EXPECTED_UPSTREAM_COMMIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_root = args.upstream_worktree.resolve()
    timing_source = upstream_root / "eval" / "run_eval.py"
    actual_upstream_commit = _git_commit(upstream_root)
    if actual_upstream_commit != args.upstream_commit:
        raise RuntimeError(
            f"upstream worktree commit {actual_upstream_commit} != {args.upstream_commit}"
        )
    source_snapshot = output_dir / "upstream_run_eval_2e21c57.py"
    shutil.copy2(timing_source, source_snapshot)

    def sample_tokens_current(module, sampler, *, n_samples, batch_size, length):
        del length
        return sample_point(
            module,
            str(sampler.kind),
            dict(sampler.arch),
            int(sampler.nfe),
            str(sampler.discretize),
            int(n_samples),
            int(batch_size),
        )

    measure_latency = _load_upstream_timer(timing_source, sample_tokens_current)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    current_commit = _git_commit(ROOT)
    metadata = {
        "experiment": "tinystories_quality_grid_batch1_latency",
        "device": args.device,
        "gpu_name": gpu_name,
        "dtype": "float32",
        "seed": args.seed,
        "length": 256,
        "batch_size": 1,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "cache_mode": {
            "default": "none",
            "M2-source-fast": "incremental_clean_prefix_kv+torch_compile",
            "M3-step100000-fast": "incremental_clean_prefix_kv+torch_compile",
        },
        "output_dir": str(output_dir),
        "current_commit": current_commit,
        "upstream_commit": actual_upstream_commit,
        "upstream_timing_source": str(source_snapshot),
        "upstream_timing_source_sha256": _sha256(source_snapshot),
        "timing_functions": ["_sync", "_measure_sequence_latency_ms"],
        "timing_code_execution": "AST nodes loaded directly from upstream source",
        "quality_sources": sorted({str(item["scores"]) for item in MODELS}),
    }
    print("[config] " + json.dumps(metadata, indent=2), flush=True)

    output_json = output_dir / "latency_quality_grid.json"
    rows = []
    if output_json.exists() and not args.no_resume:
        old = json.loads(output_json.read_text())
        if old.get("metadata", {}).get("upstream_commit") != actual_upstream_commit:
            raise RuntimeError("resume artifact has a different upstream commit")
        if old.get("metadata", {}).get("repeats") != args.repeats:
            raise RuntimeError("resume artifact has a different repeat count")
        rows = old.get("rows", [])
        print(f"[cache] resume hit: {output_json} ({len(rows)} completed points)", flush=True)
    completed = {(row["model"], row["point"]) for row in rows}

    total = sum(len(json.loads(item["dump"].read_text())["points"]) for item in MODELS)
    started_all = time.perf_counter()
    for model_idx, item in enumerate(MODELS, start=1):
        dump = json.loads(item["dump"].read_text())
        scores_doc = json.loads(item["scores"].read_text())
        score_rows = {
            row["point"]: row for row in scores_doc["rows"] if row["label"] == dump["label"]
        }
        if set(score_rows) != {point["id"] for point in dump["points"]}:
            raise RuntimeError(f"quality grid mismatch for {item['name']}")
        checkpoint = (ROOT / dump["model"]["checkpoint"]).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        arch = dict(dump["model"]["arch"])
        arch["_path"] = checkpoint
        print(
            f"[stage=model_build] model={item['name']} ({model_idx}/{len(MODELS)}) "
            f"checkpoint={checkpoint} step={dump['model']['checkpoint_step']} "
            f"sampler={dump['sampler']['kind']} block_size={dump['sampler']['block_size']}",
            flush=True,
        )
        module = build_module(arch, args.device)
        parameter_dtype = str(next(module.parameters()).dtype)
        print(f"[stage=model_ready] model={item['name']} dtype={parameter_dtype}", flush=True)

        for point in dump["points"]:
            key = (item["name"], point["id"])
            if key in completed:
                print(f"[cache] point hit: model={key[0]} point={key[1]}", flush=True)
                continue
            done = len(rows) + 1
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            sampler = OmegaConf.create({
                "kind": dump["sampler"]["kind"],
                "arch": arch,
                "nfe": point["nfe"],
                "discretize": point["discretize"],
            })
            point_start = time.perf_counter()
            print(
                f"[stage=latency] point={done}/{total} model={item['name']} "
                f"mode={point['discretize']} nfe={point['nfe']} "
                f"warmup={args.warmup_runs} repeats={args.repeats}",
                flush=True,
            )
            latency = measure_latency(
                module,
                sampler,
                length=int(dump["sampler"]["length"]),
                device=args.device,
                warmup_runs=args.warmup_runs,
                repeats=args.repeats,
            )
            quality = score_rows[point["id"]]
            row = {
                "model": item["name"],
                "label": dump["label"],
                "checkpoint": str(checkpoint),
                "checkpoint_step": dump["model"]["checkpoint_step"],
                "checkpoint_sha256_head": dump["model"]["checkpoint_sha256_head"],
                "sampler": dump["sampler"]["kind"],
                "block_size": dump["sampler"]["block_size"],
                "point": point["id"],
                "discretize": point["discretize"],
                "nfe": point["nfe"],
                "forwards_per_sequence": point["forwards_per_sequence"],
                "quality_n_samples": point["n_samples"],
                "mauve": quality["mauve"],
                "gen_ppl": quality["gen_ppl"],
                "sequence_latency_ms": latency["sequence_latency_ms"],
                "sequence_latency_p10_ms": latency["sequence_latency_p10_ms"],
                "sequence_latency_p90_ms": latency["sequence_latency_p90_ms"],
                "sequence_latency_repeats": latency["sequence_latency_repeats"],
                "latency_device": args.device,
                "latency_gpu_name": gpu_name,
                "point_benchmark_seconds": time.perf_counter() - point_start,
            }
            rows.append(row)
            completed.add(key)
            _write_outputs(output_dir, metadata, rows)
            print(
                f"[metric] model={item['name']} mode={point['discretize']} nfe={point['nfe']} "
                f"latency_median_ms={latency['sequence_latency_ms']:.3f} "
                f"p10={latency['sequence_latency_p10_ms']:.3f} "
                f"p90={latency['sequence_latency_p90_ms']:.3f} "
                f"mauve={quality['mauve']:.6f} gen_ppl={quality['gen_ppl']:.4f} "
                f"elapsed={time.perf_counter() - point_start:.1f}s",
                flush=True,
            )
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (row["model"], row["discretize"], row["nfe"]))
    _write_outputs(output_dir, metadata, rows)
    _plot(rows, output_dir, "mauve", "MAUVE", "mauve_vs_latency.png")
    _plot(rows, output_dir, "gen_ppl", "gen-PPL", "genppl_vs_latency.png")
    print(
        f"[done] rows={len(rows)} elapsed={time.perf_counter() - started_all:.1f}s "
        f"json={output_dir / 'latency_quality_grid.json'} "
        f"csv={output_dir / 'latency_quality_grid.csv'} "
        f"plots={output_dir / 'mauve_vs_latency.png'},{output_dir / 'genppl_vs_latency.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
