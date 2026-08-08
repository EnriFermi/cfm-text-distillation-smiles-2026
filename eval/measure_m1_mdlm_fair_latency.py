"""Measure M1 and MDLM under the matched BF16/max-autotune H100 protocol."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
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

from eval.dump_samples import build_module, detect_arch
from semicat.fast_inference import full_cfm_sample_fused_fast


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPSTREAM_COMMIT = "2e21c57704ab94e131e3350ffd8286632f61a7b8"


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
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _load_upstream_timer(source_path: Path, sample_tokens_fn):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    wanted = {"_sync", "_measure_sequence_latency_ms"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
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
    (output_dir / "m1_mdlm_latency.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )
    if rows:
        with (output_dir / "m1_mdlm_latency.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m1-68k-checkpoint", type=Path,
        default=ROOT / "baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--m1-200k-checkpoint", type=Path,
        default=ROOT / "logs/train/tinystories_cfm_resume144001_to200000_nfe32_20260806/checkpoints/final/step_00200000_full_state.ckpt",
    )
    parser.add_argument("--mdlm-checkpoint", type=Path, default=ROOT / "baseline/MDLM_best.pt")
    parser.add_argument(
        "--baseline-code-root", type=Path,
        default=ROOT / "external/bcfm_baselines_uploaded_20260807",
    )
    parser.add_argument("--upstream-worktree", type=Path, default=Path("/tmp/cfm-upstream-m2-latency-2e21c57"))
    parser.add_argument("--upstream-commit", default=EXPECTED_UPSTREAM_COMMIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--compile-mode", choices=("max-autotune",), default="max-autotune")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    os.environ["BCFM_INFERENCE_COMPILE_MODE"] = args.compile_mode
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code_root = args.baseline_code_root.resolve()
    sys.path.insert(0, str(code_root))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.generative_text import GenerationConfig
    from bcfm_baselines.models.mdlm_fast import generate_fast_mdlm

    upstream_root = args.upstream_worktree.resolve()
    upstream_commit = _git_commit(upstream_root)
    if upstream_commit != args.upstream_commit:
        raise RuntimeError(f"upstream commit {upstream_commit} != {args.upstream_commit}")
    timing_source = upstream_root / "eval/run_eval.py"

    checkpoints = {
        "M1-68k": args.m1_68k_checkpoint.resolve(),
        "M1-200k": args.m1_200k_checkpoint.resolve(),
        "MDLM": args.mdlm_checkpoint.resolve(),
    }
    hashes = {
        name: _sha256(path, None if name == "MDLM" else 64 << 20)
        for name, path in checkpoints.items()
    }
    metadata = {
        "experiment": "tinystories_m1_mdlm_h100_bf16_maxautotune_fair",
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": args.precision,
        "parameter_dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "compile_mode": args.compile_mode,
        "seed": args.seed,
        "batch_size": 1,
        "length": 256,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "checkpoint_hashes": hashes,
        "upstream_commit": upstream_commit,
        "upstream_timing_source": str(timing_source),
        "upstream_timing_sha256": _sha256(timing_source),
        "timing_code_execution": "AST nodes loaded directly from upstream source",
        "implementation_sha256": {
            "semicat/fast_inference.py": _sha256(ROOT / "semicat/fast_inference.py"),
            "bcfm_baselines/models/mdlm_fast.py": _sha256(
                code_root / "bcfm_baselines/models/mdlm_fast.py"
            ),
            "eval/measure_m1_mdlm_fair_latency.py": _sha256(Path(__file__)),
        },
        "output_dir": str(output_dir),
    }
    print("[config] " + json.dumps(metadata, indent=2), flush=True)

    last_diagnostics: dict = {}

    def sample_tokens(module, sampler, *, n_samples, batch_size, length):
        nonlocal last_diagnostics
        if n_samples != 1 or batch_size != 1 or length != 256:
            raise ValueError("matched latency protocol requires n=1, batch=1, length=256")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if sampler.kind == "full_cfm_fused_fast":
                last_diagnostics = {
                    "num_function_evaluations": int(sampler.nfe),
                    "sampler": "full_cfm_fused_fast",
                }
                return full_cfm_sample_fused_fast(
                    module,
                    int(sampler.nfe),
                    batch_size=1,
                    discretize="argmax",
                )
            if sampler.kind == "mdlm_fused_fast":
                result = generate_fast_mdlm(
                    module,
                    GenerationConfig(
                        length=256,
                        batch_size=1,
                        strategy="sample",
                        temperature=1.0,
                        top_k=None,
                        top_p=0.9,
                        top_p_include_boundary=False,
                        seed=args.seed,
                        num_steps=int(sampler.nfe),
                        sampling_epsilon=1e-5,
                        noise_removal=True,
                        cache_denoiser_outputs=True,
                        start_token_id=50256,
                    ),
                )
                last_diagnostics = result.diagnostics
                return result.tokens
        raise ValueError(f"unsupported sampler kind {sampler.kind!r}")

    measure_latency = _load_upstream_timer(timing_source, sample_tokens)
    output_json = output_dir / "m1_mdlm_latency.json"
    rows: list[dict] = []
    if output_json.exists() and not args.no_resume:
        previous = json.loads(output_json.read_text())
        for key in ("gpu_name", "precision", "compile_mode", "warmup_runs", "repeats", "checkpoint_hashes"):
            if previous["metadata"].get(key) != metadata.get(key):
                raise RuntimeError(f"resume metadata mismatch: {key}")
        rows = previous["rows"]
        print(f"[cache] resume={output_json} rows={len(rows)}", flush=True)
    completed = {(row["model"], row["point"]) for row in rows}
    started_all = time.perf_counter()

    for model_name in ("M1-68k", "M1-200k", "MDLM"):
        nfes = (16, 32, 64, 128, 256) if model_name == "MDLM" else (1, 2, 4, 8, 16, 32)
        pending = [nfe for nfe in nfes if (model_name, f"{'sample' if model_name == 'MDLM' else 'argmax'}_nfe{nfe}") not in completed]
        if not pending:
            print(f"[cache] model={model_name} all_points_complete", flush=True)
            continue
        print(f"[stage=model_load] model={model_name} checkpoint={checkpoints[model_name]}", flush=True)
        if model_name == "MDLM":
            module, checkpoint = load_checkpoint_model(checkpoints[model_name], args.device)
            if checkpoint.get("weights_used_for_inference") != "ema":
                raise RuntimeError("MDLM did not load EMA weights")
            del checkpoint
        else:
            checkpoint = torch.load(checkpoints[model_name], map_location="cpu", weights_only=False)
            arch = detect_arch(checkpoint)
            arch["_path"] = checkpoints[model_name]
            if arch["block_size"] is not None:
                raise RuntimeError(f"{model_name} checkpoint is not full CFM")
            del checkpoint
            module = build_module(arch, args.device)
        module = module.eval().to(dtype=torch.bfloat16)
        print(
            f"[stage=model_ready] model={model_name} parameter_dtype={next(module.parameters()).dtype} "
            f"points={pending}",
            flush=True,
        )

        for index, nfe in enumerate(pending, start=1):
            point = f"{'sample' if model_name == 'MDLM' else 'argmax'}_nfe{nfe}"
            sampler = OmegaConf.create(
                {"kind": "mdlm_fused_fast" if model_name == "MDLM" else "full_cfm_fused_fast", "nfe": nfe}
            )
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            print(
                f"[stage=latency] model={model_name} point={point} progress={index}/{len(pending)} "
                f"warmup={args.warmup_runs} repeats={args.repeats}",
                flush=True,
            )
            started = time.perf_counter()
            latency = measure_latency(
                module, sampler, length=256, device=args.device,
                warmup_runs=args.warmup_runs, repeats=args.repeats,
            )
            row = {
                "model": model_name,
                "sampler": str(sampler.kind),
                "point": point,
                "discretize": "sample" if model_name == "MDLM" else "argmax",
                "nfe": nfe,
                "backbone_calls_per_sequence": int(last_diagnostics["num_function_evaluations"]),
                "checkpoint": str(checkpoints[model_name]),
                "checkpoint_hash": hashes[model_name],
                "checkpoint_step": 100000 if model_name == "MDLM" else (68001 if model_name == "M1-68k" else 200000),
                "sequence_latency_ms": latency["sequence_latency_ms"],
                "sequence_latency_p10_ms": latency["sequence_latency_p10_ms"],
                "sequence_latency_p90_ms": latency["sequence_latency_p90_ms"],
                "sequence_latency_repeats": latency["sequence_latency_repeats"],
                "latency_device": args.device,
                "latency_gpu_name": metadata["gpu_name"],
                "precision": args.precision,
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "compile_mode": args.compile_mode,
                "batch_size": 1,
                "length": 256,
                "denoiser_cache_hits": int(last_diagnostics.get("denoiser_cache_hits", 0)),
                "point_benchmark_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            completed.add((model_name, point))
            rows.sort(key=lambda item: (item["model"], item["nfe"]))
            _write(output_dir, metadata, rows)
            print(
                f"[metric] model={model_name} point={point} median_ms={row['sequence_latency_ms']:.3f} "
                f"p10={row['sequence_latency_p10_ms']:.3f} p90={row['sequence_latency_p90_ms']:.3f} "
                f"actual_nfe={row['backbone_calls_per_sequence']} cache_hits={row['denoiser_cache_hits']} "
                f"elapsed={row['point_benchmark_seconds']:.1f}s",
                flush=True,
            )
        del module
        torch.cuda.empty_cache()

    _write(output_dir, metadata, rows)
    print(
        f"[done] rows={len(rows)} elapsed={time.perf_counter() - started_all:.1f}s "
        f"json={output_dir / 'm1_mdlm_latency.json'} csv={output_dir / 'm1_mdlm_latency.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
