"""Measure uploaded TinyStories MDLM/BD3-LM checkpoints on the CFM H100 protocol.

The baseline model and sampler implementations come from the uploaded
``baseline/code.zip`` extraction.  The synchronization/timing function is loaded
verbatim from the same upstream CFM commit used for the CFM latency curves.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UPSTREAM_COMMIT = "2e21c57704ab94e131e3350ffd8286632f61a7b8"


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
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    wanted = {"_sync", "_measure_sequence_latency_ms"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
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
    json_path = output_dir / "baseline_h100_latency.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    if rows:
        csv_path = output_dir / "baseline_h100_latency.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _point_specs() -> list[dict]:
    points = []
    for nfe in (16, 32, 64, 128, 256):
        points.append(
            {
                "model_family": "MDLM",
                "sampler": "mdlm",
                "point": f"sample_nfe{nfe}",
                "nfe": nfe,
                "forwards_per_sequence": nfe + 1,
                "block_size": None,
                "generation": {
                    "num_steps": nfe,
                    "steps_per_block": None,
                    "first_hitting": False,
                    "use_kv_cache": False,
                },
            }
        )
    for nfe in (1, 2, 4, 8, 16, 32):
        points.append(
            {
                "model_family": "BD3-LM",
                "sampler": "bd3lm_ancestral",
                "point": f"sample_nfe{nfe}",
                "nfe": nfe,
                "forwards_per_sequence": 16 * nfe + 16,
                "block_size": 16,
                "generation": {
                    "num_steps": 64,
                    "steps_per_block": nfe,
                    "first_hitting": False,
                    "use_kv_cache": True,
                },
            }
        )
    points.append(
        {
            "model_family": "BD3-LM",
            "sampler": "bd3lm_first_hitting",
            "point": "sample_nfe256",
            "nfe": 256,
            "forwards_per_sequence": 272,
            "block_size": 16,
            "generation": {
                "num_steps": 64,
                "steps_per_block": 256,
                "first_hitting": True,
                "use_kv_cache": True,
            },
        }
    )
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-code-root",
        type=Path,
        default=ROOT / "external" / "bcfm_baselines_uploaded_20260807",
    )
    parser.add_argument("--mdlm-checkpoint", type=Path, default=ROOT / "baseline/MDLM_best.pt")
    parser.add_argument("--bd3lm-checkpoint", type=Path, default=ROOT / "baseline/BD3LM_best.pt")
    parser.add_argument(
        "--upstream-worktree",
        type=Path,
        default=Path("/tmp/cfm-upstream-m2-latency-2e21c57"),
    )
    parser.add_argument("--upstream-commit", default=EXPECTED_UPSTREAM_COMMIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code_root = args.baseline_code_root.resolve()
    sys.path.insert(0, str(code_root))
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.models.generative_text import GenerationConfig

    upstream_root = args.upstream_worktree.resolve()
    actual_upstream_commit = _git_commit(upstream_root)
    if actual_upstream_commit != args.upstream_commit:
        raise RuntimeError(
            f"upstream worktree commit {actual_upstream_commit} != {args.upstream_commit}"
        )
    timing_source = upstream_root / "eval/run_eval.py"

    common_generation = {
        "length": 256,
        "batch_size": 1,
        "strategy": "sample",
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.9,
        "top_p_include_boundary": False,
        "seed": args.seed,
        "sampling_epsilon": 1e-5,
        "noise_removal": True,
        "cache_denoiser_outputs": True,
        "start_token_id": 50256,
    }

    def sample_tokens_baseline(module, sampler, *, n_samples, batch_size, length):
        if n_samples != 1 or batch_size != 1 or length != 256:
            raise ValueError("H100 latency runner is intentionally batch=1, n=1, length=256")
        generation = dict(common_generation)
        generation.update(OmegaConf.to_container(sampler.generation, resolve=True))
        result = module.generate(GenerationConfig(**generation))
        return result.tokens

    measure_latency = _load_upstream_timer(timing_source, sample_tokens_baseline)
    checkpoint_paths = {
        "MDLM": args.mdlm_checkpoint.resolve(),
        "BD3-LM": args.bd3lm_checkpoint.resolve(),
    }
    checkpoint_hashes = {name: _sha256(path) for name, path in checkpoint_paths.items()}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    metadata = {
        "experiment": "tinystories_mdlm_bd3lm_latency_h100_matched",
        "device": args.device,
        "gpu_name": gpu_name,
        "dtype": "float32",
        "seed": args.seed,
        "length": 256,
        "batch_size": 1,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "sampling": common_generation,
        "weights": "ema",
        "checkpoint_paths": {key: str(value) for key, value in checkpoint_paths.items()},
        "checkpoint_sha256": checkpoint_hashes,
        "baseline_code_root": str(code_root),
        "baseline_code_zip_sha256": _sha256(ROOT / "baseline/code.zip"),
        "upstream_commit": actual_upstream_commit,
        "upstream_timing_source": str(timing_source),
        "upstream_timing_source_sha256": _sha256(timing_source),
        "timing_code_execution": "AST nodes loaded directly from upstream source",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "output_dir": str(output_dir),
    }
    print("[config] " + json.dumps(metadata, indent=2), flush=True)

    output_json = output_dir / "baseline_h100_latency.json"
    rows: list[dict] = []
    if output_json.exists() and not args.no_resume:
        previous = json.loads(output_json.read_text())
        old_metadata = previous.get("metadata", {})
        for key in ("gpu_name", "dtype", "warmup_runs", "repeats", "checkpoint_sha256"):
            if old_metadata.get(key) != metadata.get(key):
                raise RuntimeError(f"resume metadata mismatch for {key}")
        rows = previous.get("rows", [])
        print(f"[cache] resume hit: {output_json} ({len(rows)} rows)", flush=True)
    completed = {(row["model_family"], row["sampler"], row["point"]) for row in rows}

    specs = _point_specs()
    started_all = time.perf_counter()
    for family in ("MDLM", "BD3-LM"):
        family_specs = [point for point in specs if point["model_family"] == family]
        pending = [
            point
            for point in family_specs
            if (family, point["sampler"], point["point"]) not in completed
        ]
        if not pending:
            print(f"[cache] model={family}: all {len(family_specs)} points complete", flush=True)
            continue
        checkpoint_path = checkpoint_paths[family]
        print(
            f"[stage=model_load] model={family} checkpoint={checkpoint_path} "
            f"sha256={checkpoint_hashes[family]}",
            flush=True,
        )
        module, checkpoint = load_checkpoint_model(checkpoint_path, args.device)
        config = checkpoint["config"]
        if checkpoint.get("step") != 100000:
            raise RuntimeError(f"{family} checkpoint step is {checkpoint.get('step')}, expected 100000")
        if config["dataset"]["name"] != "tinystories":
            raise RuntimeError(f"{family} checkpoint is not TinyStories")
        backbone = config["model"]["backbone"]
        expected_arch = (384, 6, 6, 256)
        actual_arch = (
            backbone["hidden_size"], backbone["num_layers"],
            backbone["num_attention_heads"], backbone["max_sequence_length"],
        )
        if actual_arch != expected_arch:
            raise RuntimeError(f"{family} architecture {actual_arch} != {expected_arch}")
        if checkpoint.get("weights_used_for_inference") != "ema":
            raise RuntimeError(f"{family} did not load EMA weights")
        parameter_dtype = str(next(module.parameters()).dtype)
        del checkpoint
        print(
            f"[stage=model_ready] model={family} step=100000 arch=384x6 "
            f"parameter_dtype={parameter_dtype} points={len(pending)}",
            flush=True,
        )

        for index, point in enumerate(pending, start=1):
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            sampler = OmegaConf.create({"generation": point["generation"]})
            started = time.perf_counter()
            print(
                f"[stage=latency] model={family} point={point['point']} "
                f"sampler={point['sampler']} nfe={point['nfe']} "
                f"progress={index}/{len(pending)} warmup={args.warmup_runs} "
                f"repeats={args.repeats}",
                flush=True,
            )
            latency = measure_latency(
                module,
                sampler,
                length=256,
                device=args.device,
                warmup_runs=args.warmup_runs,
                repeats=args.repeats,
            )
            row = {
                "model_family": family,
                "model_variant": "best_canonical_100k",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hashes[family],
                "checkpoint_step": 100000,
                "weights": "ema",
                "sampler": point["sampler"],
                "block_size": point["block_size"],
                "point": point["point"],
                "discretize": "sample",
                "nfe": point["nfe"],
                "forwards_per_sequence": point["forwards_per_sequence"],
                "sequence_latency_ms": latency["sequence_latency_ms"],
                "sequence_latency_p10_ms": latency["sequence_latency_p10_ms"],
                "sequence_latency_p90_ms": latency["sequence_latency_p90_ms"],
                "sequence_latency_repeats": latency["sequence_latency_repeats"],
                "latency_device": args.device,
                "latency_gpu_name": gpu_name,
                "dtype": "float32",
                "batch_size": 1,
                "length": 256,
                "protocol_id": f"upstream_{actual_upstream_commit[:8]}_h100_fp32_b1_l256_v1",
                "point_benchmark_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            completed.add((family, point["sampler"], point["point"]))
            rows.sort(key=lambda item: (item["model_family"], item["sampler"], item["nfe"]))
            _write_outputs(output_dir, metadata, rows)
            point_dir = output_dir / f"{family.lower().replace('-', '')}_{point['sampler']}_{point['point']}"
            point_dir.mkdir(parents=True, exist_ok=True)
            (point_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
            print(
                f"[metric] model={family} point={point['point']} "
                f"latency_median_ms={latency['sequence_latency_ms']:.3f} "
                f"p10={latency['sequence_latency_p10_ms']:.3f} "
                f"p90={latency['sequence_latency_p90_ms']:.3f} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        del module
        torch.cuda.empty_cache()

    _write_outputs(output_dir, metadata, rows)
    print(
        f"[done] rows={len(rows)} elapsed={time.perf_counter() - started_all:.1f}s "
        f"json={output_dir / 'baseline_h100_latency.json'} "
        f"csv={output_dir / 'baseline_h100_latency.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
