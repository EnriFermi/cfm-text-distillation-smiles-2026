"""Build the final provenance and token-level M2 comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import transformers


CURRENT_SOURCE_FILES = (
    "eval/dump_samples.py",
    "block/sampling.py",
    "block/block_dit.py",
    "block/block_semicat.py",
    "block/mask.py",
    "semicat/net/duo.py",
    "semicat/models/semicat.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_output(root: Path, *args: str, binary: bool = False):
    return subprocess.check_output(
        ["git", *args], cwd=root, text=not binary
    )


def load_point(path: Path) -> tuple[dict, dict, np.ndarray]:
    dump = json.loads(path.read_text())
    if len(dump["points"]) != 1:
        raise ValueError(f"expected exactly one point in {path}")
    point = dump["points"][0]
    tokens = np.load(path.with_suffix(".tokens.npz"))[point["id"]]
    return dump, point, tokens


def junit_summary(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        raise ValueError(f"no testsuite in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "seconds": float(suite.attrib["time"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--old-current", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--paired-judge", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    current_dump, current_point, current = load_point(args.current)
    upstream_dump, upstream_point, upstream = load_point(args.upstream)
    old_current = np.load(args.old_current.with_suffix(".tokens.npz"))[
        current_point["id"]
    ]
    if current.shape != upstream.shape:
        raise ValueError(f"shape mismatch: {current.shape} versus {upstream.shape}")
    if current.ndim != 2:
        raise ValueError(f"expected rank-2 token arrays, got {current.shape}")

    n_samples, length = current.shape
    block_size = current_dump["sampler"]["block_size"]
    if length % block_size:
        raise ValueError("length is not divisible by block size")
    equal = current == upstream
    sequence_hamming = np.sum(~equal, axis=1)
    exact_sequences = np.all(equal, axis=1)
    current_texts = current_point["texts"]
    upstream_texts = upstream_point["texts"]
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    current_decoded = tokenizer.batch_decode(current, skip_special_tokens=True)
    upstream_decoded = tokenizer.batch_decode(upstream, skip_special_tokens=True)

    per_block = []
    first_divergence = np.full(n_samples, -1, dtype=np.int64)
    for block in range(length // block_size):
        lo, hi = block * block_size, (block + 1) * block_size
        block_equal = equal[:, lo:hi]
        differs_here = np.any(~block_equal, axis=1)
        first_divergence[(first_divergence < 0) & differs_here] = block
        per_block.append(
            {
                "block": block,
                "positions": [lo, hi - 1],
                "matching_tokens": int(block_equal.sum()),
                "total_tokens": int(block_equal.size),
                "token_agreement": float(block_equal.mean()),
                "exact_block_sequences": int(np.all(block_equal, axis=1).sum()),
                "exact_block_sequence_rate": float(np.all(block_equal, axis=1).mean()),
            }
        )

    current_commit = git_output(repo_root, "rev-parse", "HEAD").strip()
    current_sources = {}
    for rel in CURRENT_SOURCE_FILES:
        worktree_bytes = (repo_root / rel).read_bytes()
        commit_bytes = git_output(repo_root, "show", f"{current_commit}:{rel}", binary=True)
        current_sources[rel] = {
            "sha256": bytes_sha256(worktree_bytes),
            "git_blob_sha256": bytes_sha256(commit_bytes),
            "matches_commit": worktree_bytes == commit_bytes,
        }

    upstream_commit = upstream_dump["implementation"]["source_commit"]
    upstream_sources_verified = {}
    for rel, recorded_sha in upstream_dump["implementation"]["source_files"].items():
        commit_bytes = git_output(
            args.upstream_root, "show", f"{upstream_commit}:{rel}", binary=True
        )
        actual_sha = bytes_sha256(commit_bytes)
        upstream_sources_verified[rel] = {
            "recorded_sha256": recorded_sha,
            "git_sha256": actual_sha,
            "matches_git": actual_sha == recorded_sha,
        }

    checkpoint_sha = file_sha256(args.checkpoint)
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_hparams = checkpoint_payload.get("hyper_parameters") or {}
    checkpoint_metadata = {
        "global_step": checkpoint_payload.get("global_step"),
        "trained_block_size": checkpoint_hparams.get("block_size"),
        "prior_type": checkpoint_hparams.get("prior_type"),
        "sd_type": checkpoint_hparams.get("sd_type"),
    }
    del checkpoint_payload
    scores = json.loads(args.scores.read_text())
    score_by_label = {row["label"]: row for row in scores["rows"]}
    current_scores = score_by_label[current_dump["label"]]
    upstream_scores = score_by_label[upstream_dump["label"]]
    metric_names = (
        "token_entropy",
        "seq_rep_2",
        "distinct_2",
        "seq_rep_3",
        "distinct_3",
        "seq_rep_4",
        "distinct_4",
        "cross_sample_overlap_4",
        "js_1gram",
        "js_2gram",
        "mauve",
        "frontier_integral",
        "gen_ppl",
    )
    score_comparison = {
        name: {
            "current": current_scores[name],
            "upstream": upstream_scores[name],
            "upstream_minus_current": upstream_scores[name] - current_scores[name],
        }
        for name in metric_names
    }

    first_divergence_histogram = {
        "exact": int(np.sum(first_divergence < 0)),
        **{
            str(block): int(np.sum(first_divergence == block))
            for block in range(length // block_size)
        },
    }
    quantiles = np.quantile(sequence_hamming, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    paired_judge = json.loads(args.paired_judge.read_text())
    current_junit = junit_summary(args.out.parent / "current_tests.xml")
    upstream_junit = junit_summary(args.out.parent / "upstream_tests.xml")
    equivalence_probes = {}
    for label, filename in (
        ("fp32_matmul_highest", "probe_results.json"),
        ("fp32_matmul_high_production", "probe_results_high.json"),
    ):
        probe_path = args.out.parent / filename
        probe = json.loads(probe_path.read_text())
        equivalence_probes[label] = {
            "path": str(probe_path.resolve()),
            "sha256": file_sha256(probe_path),
            "matmul_precision": probe["environment"]["float32_matmul_precision"],
            "strict_load_ok": not any(
                probe["model"][key]
                for key in (
                    "current_strict_load_missing_keys",
                    "current_strict_load_unexpected_keys",
                    "remote_strict_load_missing_keys",
                    "remote_strict_load_unexpected_keys",
                )
            ),
            "logit_probe": probe["logit_probe"]["aggregate"],
            "e2e_probe": probe["e2e_probe"]["aggregate"],
            "supported": probe["conclusion"]["supported"],
        }
    first_exact_index = int(np.flatnonzero(exact_sequences)[0])
    first_divergent_index = int(np.flatnonzero(~exact_sequences)[0])
    p95_index = int(np.argmin(np.abs(sequence_hamming - quantiles[5])))
    max_index = int(np.argmax(sequence_hamming))
    qualitative_indices = dict.fromkeys(
        (first_exact_index, first_divergent_index, p95_index, max_index)
    )
    qualitative_examples = [
        {
            "index": index,
            "hamming_tokens": int(sequence_hamming[index]),
            "current": current_texts[index],
            "upstream": upstream_texts[index],
        }
        for index in qualitative_indices
    ]

    current_command = (
        "/home/coder/.conda/envs/semicat/bin/python -m eval.dump_samples "
        "--checkpoint baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt "
        "--sampler block_causal --infer-block-size 16 --nfe 4 "
        "--discretize argmax --n-samples 512 --batch-size 64 --seed 0 "
        "--device cuda --label current_m2_blockdit_masked "
        "--out artifacts/m2_impl_comparison_20260731/current_m2_nfe4.json"
    )
    upstream_command = (
        "/home/coder/.conda/envs/semicat/bin/python "
        "artifacts/m2_impl_comparison_20260731/run_upstream_exact.py "
        "--source-root /home/coder/tmp/cfm-m2-upstream-2e21c57 "
        "--checkpoint baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt "
        "--out artifacts/m2_impl_comparison_20260731/upstream_m2_nfe4.json "
        "--device cuda --seed 0 --length 256 --block-size 16 --nfe 4 "
        "--n-samples 512 --batch-size 64"
    )

    payload = {
        "scope": "TinyStories M2 comparison at argmax NFE=4 only",
        "protocol": {
            "dataset": "tinystories",
            "length": length,
            "block_size": block_size,
            "nfe_per_block": current_point["nfe"],
            "schedule": "uniform",
            "discretize": current_point["discretize"],
            "prior": "gaussian",
            "seed": current_dump["sampler"]["seed"],
            "n_samples": n_samples,
            "batch_size": 64,
            "dtype": upstream_dump["implementation"]["dtype"],
            "device": "cuda",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_metadata": checkpoint_metadata,
            "commands": {
                "current": current_command,
                "upstream": upstream_command,
                "scorer": (
                    "python -m eval.score_texts --dumps current_m2_nfe4.json "
                    "upstream_m2_nfe4.json --reference dumps/gold_tinystories.json "
                    "--metrics entropy rep diversity ngram mauve gen_ppl "
                    "--judge EleutherAI/gpt-j-6B --judge-batch-size 8 "
                    "--judge-dtype float32 --featurize-model gpt2-large "
                    "--context-size 256 --device cuda --device-id 0"
                ),
            },
        },
        "provenance": {
            "current": {
                "commit": current_commit,
                "dump_git_commit": current_dump["git_commit"],
                "relevant_sources": current_sources,
                "all_relevant_sources_match_commit": all(
                    entry["matches_commit"] for entry in current_sources.values()
                ),
            },
            "upstream": {
                "requested_remote_name": "m2-infer-experiment",
                "resolved_ref": "origin/feature/m2-infer-experiments",
                "commit": upstream_commit,
                "worktree": str(args.upstream_root.resolve()),
                "worktree_clean": not bool(
                    git_output(args.upstream_root, "status", "--porcelain").strip()
                ),
                "source_verification": upstream_sources_verified,
                "all_recorded_sources_match_git": all(
                    entry["matches_git"] for entry in upstream_sources_verified.values()
                ),
                "imported_sources": upstream_dump["implementation"]["imported_sources"],
            },
        },
        "validity_checks": {
            "same_checkpoint_sha256": (
                checkpoint_sha
                == upstream_dump["implementation"]["checkpoint_sha256"]
            ),
            "same_shape": list(current.shape) == list(upstream.shape),
            "current_reproduces_prior_dump_bitwise": bool(
                np.array_equal(current, old_current)
            ),
            "current_unique_sequences": int(np.unique(current, axis=0).shape[0]),
            "upstream_unique_sequences": int(np.unique(upstream, axis=0).shape[0]),
            "current_token_range": [int(current.min()), int(current.max())],
            "upstream_token_range": [int(upstream.min()), int(upstream.max())],
            "current_json_decode_mismatches": int(
                sum(a != b for a, b in zip(current_texts, current_decoded))
            ),
            "upstream_json_decode_mismatches": int(
                sum(a != b for a, b in zip(upstream_texts, upstream_decoded))
            ),
        },
        "token_comparison": {
            "array_sha256_raw_c_order": {
                "current": bytes_sha256(np.ascontiguousarray(current).tobytes()),
                "upstream": bytes_sha256(np.ascontiguousarray(upstream).tobytes()),
            },
            "matching_tokens": int(equal.sum()),
            "total_tokens": int(equal.size),
            "differing_tokens": int((~equal).sum()),
            "token_agreement": float(equal.mean()),
            "exact_sequences": int(exact_sequences.sum()),
            "total_sequences": n_samples,
            "exact_sequence_rate": float(exact_sequences.mean()),
            "mean_hamming_tokens_per_sequence": float(sequence_hamming.mean()),
            "hamming_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("min", "p25", "median", "p75", "p90", "p95", "p99", "max"),
                    quantiles,
                )
            },
            "first_divergence_block_histogram": first_divergence_histogram,
            "per_block": per_block,
        },
        "qualitative_review": {
            "identical_decoded_text_pairs": int(
                sum(a == b for a, b in zip(current_texts, upstream_texts))
            ),
            "empty_current_texts": int(sum(not text.strip() for text in current_texts)),
            "empty_upstream_texts": int(sum(not text.strip() for text in upstream_texts)),
            "examples": qualitative_examples,
            "review_note": (
                "Both arms contain visibly incoherent, repetitive, and mojibake-heavy "
                "TinyStories text. The comparison establishes implementation behavior, "
                "not that this checkpoint/NFE point is a good generation baseline."
            ),
        },
        "quality_metrics": {
            "scorer": scores,
            "comparison": score_comparison,
            "paired_gpt_j": paired_judge,
        },
        "execution": {
            "current": {
                "algorithm": "full doubled-sequence masked BlockDIT",
                "flow_forwards_per_sequence": current_point["forwards_per_sequence"],
                "cache_encode_forwards": 0,
                "sampling_seconds": current_point["sampling_seconds"],
            },
            "upstream": {
                "algorithm": "current-block forward with clean-prefix KV cache",
                "flow_forwards_per_sequence": upstream_point["flow_forwards_per_sequence"],
                "cache_encode_forwards": upstream_point["cache_encode_forwards"],
                "sampling_seconds": upstream_point["sampling_seconds"],
            },
            "observed_single_run_speed_ratio_current_over_upstream": (
                current_point["sampling_seconds"] / upstream_point["sampling_seconds"]
            ),
            "timing_caveat": (
                "Synchronized same-GPU end-to-end sampling timings with model load excluded, "
                "but only one timed run per implementation; not a steady-state benchmark."
            ),
        },
        "tests": {
            "current": current_junit,
            "upstream": upstream_junit,
            "caveat": (
                "These revision-local suites passed, but they are not by themselves "
                "a cross-revision actual-checkpoint equivalence test."
            ),
        },
        "same_state_equivalence_probes": equivalence_probes,
        "interpretation_boundary": (
            "Established only for argmax at NFE=4, seed 0. This does not establish "
            "equivalence for categorical sampling, other NFE values, seeds, or checkpoints."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    current_manifest = {
        "command": current_command,
        "commit": current_commit,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metadata": checkpoint_metadata,
        "device": "cuda",
        "dtype": "torch.float32",
        "seed": 0,
        "length": length,
        "block_size": block_size,
        "nfe": current_point["nfe"],
        "schedule": "uniform",
        "discretize": current_point["discretize"],
        "n_samples": n_samples,
        "batch_size": 64,
        "source_files": {
            rel: entry["sha256"] for rel, entry in current_sources.items()
        },
        "relevant_sources_match_commit": all(
            entry["matches_commit"] for entry in current_sources.values()
        ),
        "output": str(args.current.resolve()),
    }
    (args.out.parent / "current_run_manifest.json").write_text(
        json.dumps(current_manifest, indent=2) + "\n"
    )
    (args.out.parent / "upstream_run_manifest.json").write_text(
        json.dumps(upstream_dump["implementation"], indent=2) + "\n"
    )
    print(
        f"[done] {args.out} agreement={equal.mean():.6%} "
        f"exact_sequences={exact_sequences.sum()}/{n_samples} "
        f"speed_ratio={payload['execution']['observed_single_run_speed_ratio_current_over_upstream']:.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
