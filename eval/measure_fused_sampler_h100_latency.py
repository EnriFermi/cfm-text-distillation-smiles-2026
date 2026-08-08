"""Measure fused-cache M2, M3, and BD3-LM latency on the matched H100 protocol."""

from __future__ import annotations

import argparse
import ast
import csv
import contextlib
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from eval.dump_samples import build_module, sample_point


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPSTREAM_COMMIT = "2e21c57704ab94e131e3350ffd8286632f61a7b8"
STEPS = (1, 2, 4, 8, 16, 32)


def _sha256(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit
        while True:
            size = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def _load_upstream_timer(source_path: Path, sample_tokens_fn):
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    wanted = {"_sync", "_measure_sequence_latency_ms"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("upstream timing functions are missing")
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


def _write(output_dir: Path, metadata: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fused_sampler_latency.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )
    if rows:
        with (output_dir / "fused_sampler_latency.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


def _old_latencies() -> dict[tuple[str, str, str], float]:
    old: dict[tuple[str, str, str], float] = {}
    cfm_path = ROOT / "artifacts/tinystories_latency_quality_grid_20260806/latency_quality_grid.json"
    for row in json.loads(cfm_path.read_text())["rows"]:
        if row["model"] == "M2-source-fast":
            old[("M2", row["discretize"], row["point"])] = row["sequence_latency_ms"]
        elif row["model"] == "M3-step100000-fast":
            old[("M3", row["discretize"], row["point"])] = row["sequence_latency_ms"]
    bd3_path = ROOT / "artifacts/tinystories_mdlm_bd3lm_latency_h100_20260807/baseline_h100_latency.json"
    for row in json.loads(bd3_path.read_text())["rows"]:
        if row["model_family"] == "BD3-LM":
            old[("BD3-LM", "sample", row["point"])] = row["sequence_latency_ms"]
    return old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m2-checkpoint", type=Path,
        default=ROOT / "baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--m3-checkpoint", type=Path,
        default=ROOT / "logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt",
    )
    parser.add_argument("--bd3-checkpoint", type=Path, default=ROOT / "baseline/BD3LM_best.pt")
    parser.add_argument(
        "--baseline-code-root", type=Path,
        default=ROOT / "external/bcfm_baselines_uploaded_20260807",
    )
    parser.add_argument(
        "--upstream-worktree", type=Path,
        default=Path("/tmp/cfm-upstream-m2-latency-2e21c57"),
    )
    parser.add_argument("--upstream-commit", default=EXPECTED_UPSTREAM_COMMIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32_tf32", "bf16"),
        default="fp32_tf32",
        help=(
            "Shared compute policy for M2, M3, and BD3-LM. fp32_tf32 keeps "
            "FP32 parameters/activations and enables TF32 GEMMs; bf16 casts "
            "all three models to BF16 and runs them under BF16 autocast."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("reduce-overhead", "max-autotune"),
        default="reduce-overhead",
        help="Shared TorchInductor mode for the CFM and BD3-LM block scorers.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    # Set this before recording metadata or constructing any model.  Semicat's
    # module constructor also selects ``high``; making it explicit here keeps
    # resumed runs and BD3-only runs on the identical shared policy.
    torch.set_float32_matmul_precision("high")
    os.environ["BCFM_INFERENCE_COMPILE_MODE"] = args.compile_mode

    output_dir = args.output_dir.resolve()
    upstream_root = args.upstream_worktree.resolve()
    upstream_commit = _git_commit(upstream_root)
    if upstream_commit != args.upstream_commit:
        raise RuntimeError(f"upstream commit {upstream_commit} != {args.upstream_commit}")
    timing_source = upstream_root / "eval/run_eval.py"
    code_root = args.baseline_code_root.resolve()
    sys.path.insert(0, str(code_root))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.bd3lm_fast import generate_fast_bd3
    from bcfm_baselines.models.generative_text import GenerationConfig

    def sample_tokens_optimized(module, sampler, *, n_samples, batch_size, length):
        if n_samples != 1 or batch_size != 1 or length != 256:
            raise ValueError("latency protocol is fixed at batch=1, n=1, length=256")
        precision_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if args.precision == "bf16"
            else contextlib.nullcontext()
        )
        with precision_context:
            kind = str(sampler.kind)
            if kind == "block_causal_fused_fast":
                return sample_point(
                    module, kind, dict(sampler.arch), int(sampler.steps_per_block),
                    str(sampler.discretize), 1, 1,
                )
            if kind in {"bd3lm_fused_ancestral", "bd3lm_fused_first_hitting"}:
                generation = GenerationConfig(
                    length=256,
                    batch_size=1,
                    strategy="sample",
                    temperature=1.0,
                    top_k=None,
                    top_p=0.9,
                    top_p_include_boundary=False,
                    seed=args.seed,
                    num_steps=64,
                    steps_per_block=int(sampler.steps_per_block),
                    first_hitting=kind.endswith("first_hitting"),
                    use_kv_cache=True,
                    start_token_id=50256,
                )
                return generate_fast_bd3(module, generation).tokens
            raise ValueError(f"unknown optimized sampler {kind}")

    timer = _load_upstream_timer(timing_source, sample_tokens_optimized)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    checkpoints = {
        "M2": args.m2_checkpoint.resolve(),
        "M3": args.m3_checkpoint.resolve(),
        "BD3-LM": args.bd3_checkpoint.resolve(),
    }
    hashes = {
        name: _sha256(path, None if name == "BD3-LM" else 64 << 20)
        for name, path in checkpoints.items()
    }
    metadata = {
        "experiment": "tinystories_fused_sampler_h100_latency",
        "device": args.device,
        "gpu_name": gpu_name,
        "precision": args.precision,
        "parameter_dtype": "bfloat16" if args.precision == "bf16" else "float32",
        "compute_dtype": "bfloat16" if args.precision == "bf16" else "float32_with_tf32_gemms",
        "compile_mode": args.compile_mode,
        "seed": args.seed,
        "batch_size": 1,
        "length": 256,
        "block_size": 16,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "checkpoint_hashes": hashes,
        "current_commit": _git_commit(ROOT),
        "implementation_source_sha256": {
            "block/fast_inference.py": _sha256(ROOT / "block/fast_inference.py"),
            "block/sampling.py": _sha256(ROOT / "block/sampling.py"),
            "bcfm_baselines/models/bd3lm_fast.py": _sha256(
                code_root / "bcfm_baselines/models/bd3lm_fast.py"
            ),
            "eval/measure_fused_sampler_h100_latency.py": _sha256(Path(__file__)),
        },
        "upstream_commit": upstream_commit,
        "upstream_timing_source": str(timing_source),
        "upstream_timing_sha256": _sha256(timing_source),
        "timing_code_execution": "AST nodes loaded directly from upstream source",
        "optimization": {
            "M2/M3": "current-block CUDA graphs + dynamic clean KV + fused clean-prev/first-noisy transition",
            "BD3-LM": "current-block CUDA graphs + dynamic clean KV + fused transition + no CPU sync diagnostics",
            "separate_cache_maintenance_calls": 0,
        },
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "output_dir": str(output_dir),
    }
    print("[config] " + json.dumps(metadata, indent=2), flush=True)
    old = _old_latencies()
    rows: list[dict] = []
    output_json = output_dir / "fused_sampler_latency.json"
    if output_json.exists() and not args.no_resume:
        previous = json.loads(output_json.read_text())
        for key in (
            "gpu_name", "precision", "parameter_dtype", "compute_dtype", "compile_mode", "seed",
            "warmup_runs", "repeats", "checkpoint_hashes",
        ):
            if previous["metadata"].get(key) != metadata.get(key):
                raise RuntimeError(f"resume metadata mismatch: {key}")
        rows = previous["rows"]
        print(f"[cache] resume={output_json} rows={len(rows)}", flush=True)
    completed = {(row["model"], row["point"]) for row in rows}
    started_all = time.perf_counter()

    for model_name in ("M2", "M3", "BD3-LM"):
        if model_name == "BD3-LM":
            specs = [
                ("sample", f"sample_nfe{steps}", steps, "bd3lm_fused_ancestral", 16 * steps)
                for steps in STEPS
            ] + [("sample", "sample_nfe256", 256, "bd3lm_fused_first_hitting", 256)]
        else:
            specs = [
                (mode, f"{mode}_nfe{steps}", steps, "block_causal_fused_fast", 16 * steps)
                for mode in ("argmax", "sample") for steps in STEPS
            ]
        pending = [spec for spec in specs if (model_name, spec[1]) not in completed]
        if not pending:
            print(f"[cache] model={model_name} all_points_complete", flush=True)
            continue
        print(
            f"[stage=model_load] model={model_name} checkpoint={checkpoints[model_name]} "
            f"hash={hashes[model_name]} points={len(pending)}",
            flush=True,
        )
        if model_name == "BD3-LM":
            module, payload = load_checkpoint_model(checkpoints[model_name], args.device)
            if payload.get("step") != 100000 or payload["config"]["dataset"]["name"] != "tinystories":
                raise RuntimeError("unexpected BD3-LM checkpoint provenance")
            checkpoint_step = payload["step"]
            del payload
            arch = None
        else:
            checkpoint = torch.load(checkpoints[model_name], map_location="cpu", weights_only=False)
            from eval.dump_samples import detect_arch
            arch = detect_arch(checkpoint)
            arch["_path"] = checkpoints[model_name]
            if not arch["block_size"]:
                arch["block_size"] = 16
            checkpoint_step = checkpoint.get("global_step")
            del checkpoint
            module = build_module(arch, args.device)
        if args.precision == "bf16":
            module = module.to(dtype=torch.bfloat16)
        print(
            f"[stage=model_ready] model={model_name} step={checkpoint_step} "
            f"parameter_dtype={next(module.parameters()).dtype} "
            f"compute_dtype={metadata['compute_dtype']} "
            f"compile_mode={args.compile_mode} "
            f"matmul_precision={torch.get_float32_matmul_precision()} "
            f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True,
        )

        for index, (mode, point, steps_per_block, sampler_kind, total_nfe) in enumerate(pending, 1):
            effective_steps_per_block = (
                16 if sampler_kind == "bd3lm_fused_first_hitting" else steps_per_block
            )
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            sampler = OmegaConf.create({
                "kind": sampler_kind,
                "arch": arch,
                "steps_per_block": steps_per_block,
                "discretize": mode,
            })
            point_start = time.perf_counter()
            print(
                f"[stage=latency] model={model_name} point={point} progress={index}/{len(pending)} "
                f"steps_per_block={effective_steps_per_block} "
                f"configured_steps_per_block={steps_per_block} "
                f"total_denoising_nfe={total_nfe} "
                f"warmup={args.warmup_runs} repeats={args.repeats}", flush=True,
            )
            latency = timer(
                module, sampler, length=256, device=args.device,
                warmup_runs=args.warmup_runs, repeats=args.repeats,
            )
            old_ms = old.get((model_name, mode, point))
            row = {
                "model": model_name,
                "sampler": sampler_kind,
                "point": point,
                "discretize": mode,
                "steps_per_block": effective_steps_per_block,
                "total_denoising_nfe": total_nfe,
                "backbone_calls_per_sequence": total_nfe,
                "separate_cache_calls": 0,
                "checkpoint": str(checkpoints[model_name]),
                "checkpoint_step": checkpoint_step,
                "checkpoint_hash": hashes[model_name],
                "sequence_latency_ms": latency["sequence_latency_ms"],
                "sequence_latency_p10_ms": latency["sequence_latency_p10_ms"],
                "sequence_latency_p90_ms": latency["sequence_latency_p90_ms"],
                "sequence_latency_repeats": latency["sequence_latency_repeats"],
                "previous_fast_latency_ms": old_ms,
                "speedup_vs_previous_fast": (
                    old_ms / latency["sequence_latency_ms"] if old_ms is not None else None
                ),
                "latency_device": args.device,
                "latency_gpu_name": gpu_name,
                "precision": args.precision,
                "parameter_dtype": str(next(module.parameters()).dtype).removeprefix("torch."),
                "compute_dtype": metadata["compute_dtype"],
                "compile_mode": args.compile_mode,
                "batch_size": 1,
                "length": 256,
                "point_benchmark_seconds": time.perf_counter() - point_start,
            }
            rows.append(row)
            completed.add((model_name, point))
            rows.sort(key=lambda item: (item["model"], item["discretize"], item["total_denoising_nfe"]))
            _write(output_dir, metadata, rows)
            print(
                f"[metric] model={model_name} point={point} median_ms={latency['sequence_latency_ms']:.3f} "
                f"p10={latency['sequence_latency_p10_ms']:.3f} p90={latency['sequence_latency_p90_ms']:.3f} "
                f"speedup={row['speedup_vs_previous_fast'] if row['speedup_vs_previous_fast'] is not None else 'n/a'} "
                f"elapsed={time.perf_counter() - point_start:.1f}s", flush=True,
            )
        del module
        torch.cuda.empty_cache()

    _write(output_dir, metadata, rows)
    print(
        f"[done] rows={len(rows)} elapsed={time.perf_counter() - started_all:.1f}s "
        f"json={output_dir / 'fused_sampler_latency.json'} "
        f"csv={output_dir / 'fused_sampler_latency.csv'}", flush=True,
    )


if __name__ == "__main__":
    main()
