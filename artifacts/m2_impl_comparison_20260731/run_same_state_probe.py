#!/usr/bin/env python3
"""Reproduce the same-state comparison of two M2 execution implementations.

This is an evidence-only script. It does not import or write into the remote
working tree. Instead, it loads the immutable remote sources with ``git show``:

* current arm: 07ffd55 ``BlockDIT`` + full masked 2L ``block_causal_sample``;
* remote arm: 2e21c57 ``DuoDIT`` + clean-prefix KV-cached ``blockwise_sample``.

Both arms strict-load the same TinyStories M1 network state. The script first
compares single-block logits through matching SDPA paths, then compares final
tokens using the production current FlexAttention path versus remote cached
SDPA. Results and complete provenance are written to JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURRENT_COMMIT = "07ffd555dbbbd544b22933b5a23e9ab64d633dea"
REMOTE_COMMIT = "2e21c57704ab94e131e3350ffd8286632f61a7b8"
REMOTE_BRANCH_ALIAS = "origin/feature/m2-infer-experiments"
EXPECTED_CHECKPOINT_SHA256 = (
    "a7a85b5626c9bd2c743a8773827b1ea1f24de831186490b934ceecab4a41ba6f"
)

DEFAULT_CHECKPOINT = Path(
    "baseline/tinystories/NEW/tinystories_a100/checkpoints/best.ckpt"
)
DEFAULT_OUTPUT = Path(
    "artifacts/m2_impl_comparison_20260731/probe_results.json"
)

CURRENT_SOURCE_PATHS = (
    "block/block_dit.py",
    "block/sampling.py",
    "block/mask.py",
    "semicat/net/duo.py",
)
REMOTE_SOURCE_PATHS = (
    "semicat/net/duo.py",
    "block/sampling.py",
)

MODEL_CONFIG = {
    "vocab_size": 50_257,
    "hidden_size": 384,
    "cond_dim": 128,
    "n_blocks": 6,
    "n_heads": 6,
    "dropout": 0.1,
    "length": 256,
    "embed_type": "rms",
}
BLOCK_SIZE = 16
LOGIT_PROBE_BLOCKS = (0, 1, 8, 15)
LOGIT_PROBE_SEED = 173
LOGIT_PROBE_S = 0.25
LOGIT_PROBE_T = 0.75
E2E_NFE = (1, 2)
E2E_DISCRETIZE = ("argmax", "sample")
E2E_BATCH_SIZE = 2
E2E_SEED_BASE = 9000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high"),
        default="highest",
        help="PyTorch float32 matmul precision; production SemicatModule uses high.",
    )
    parser.add_argument(
        "--allow-current-revision-mismatch",
        action="store_true",
        help="Record, rather than reject, a current HEAD other than 07ffd55.",
    )
    parser.add_argument(
        "--allow-checkpoint-hash-mismatch",
        action="store_true",
        help="Record, rather than reject, a checkpoint other than the audited M1.",
    )
    return parser.parse_args()


def run_git(repo_root: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=text
    ).strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(repo_root: Path, revision: str, path: str) -> dict[str, Any]:
    payload = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=repo_root
    )
    return {
        "revision": revision,
        "path": path,
        "git_blob": run_git(repo_root, "rev-parse", f"{revision}:{path}"),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }


def load_git_module(
    repo_root: Path,
    revision: str,
    path: str,
    module_name: str,
) -> types.ModuleType:
    source = run_git(repo_root, "show", f"{revision}:{path}")
    assert isinstance(source, str)
    module = types.ModuleType(module_name)
    module.__file__ = f"git://{revision}/{path}"
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def normalize_net_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in raw_state.items():
        if not key.startswith("net."):
            continue
        key = key[len("net.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        state[key] = value
    if not state:
        raise RuntimeError("checkpoint has no net.* state")
    return state


def reset_rng(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SamplerModule:
    """Minimal exact interface used by both sampling functions."""

    def __init__(self, torch_module: Any, net: Any):
        self._torch = torch_module
        self.net = net
        self.in_shape = (MODEL_CONFIG["length"], MODEL_CONFIG["vocab_size"])

    @property
    def device(self) -> Any:
        return next(self.net.parameters()).device

    def prior(self, shape: tuple[int, ...], device: Any) -> Any:
        return self._torch.randn(shape, device=device)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    current_head = str(run_git(repo_root, "rev-parse", "HEAD"))
    remote_alias_target = str(
        run_git(repo_root, "rev-parse", REMOTE_BRANCH_ALIAS)
    )

    print("[config] experiment=m2_same_state_implementation_probe", flush=True)
    print(f"[config] repo={repo_root}", flush=True)
    print(f"[config] current={current_head}", flush=True)
    print(f"[config] remote={REMOTE_COMMIT}", flush=True)
    print(f"[config] checkpoint={checkpoint_path}", flush=True)
    print(
        f"[config] device={args.device} dtype=float32 "
        f"matmul_precision={args.matmul_precision}",
        flush=True,
    )
    print(
        f"[config] logit_seed={LOGIT_PROBE_SEED} "
        f"e2e_seeds={[E2E_SEED_BASE + n for n in E2E_NFE]}",
        flush=True,
    )
    print("[config] cache_mode=remote_clean_prefix_kv", flush=True)
    print(f"[config] output={output_path}", flush=True)

    print("[stage] validating immutable inputs", flush=True)
    if current_head != CURRENT_COMMIT and not args.allow_current_revision_mismatch:
        raise RuntimeError(
            f"current HEAD is {current_head}, expected {CURRENT_COMMIT}; "
            "checkout the audited revision or pass the explicit override"
        )
    if remote_alias_target != REMOTE_COMMIT:
        print(
            f"[warning] {REMOTE_BRANCH_ALIAS} now resolves to {remote_alias_target}; "
            f"the probe still loads immutable {REMOTE_COMMIT}",
            flush=True,
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256
        and not args.allow_checkpoint_hash_mismatch
    ):
        raise RuntimeError(
            f"checkpoint SHA-256 is {checkpoint_sha256}, "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )

    tracked_source_clean = (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                CURRENT_COMMIT,
                "--",
                *CURRENT_SOURCE_PATHS,
            ],
            cwd=repo_root,
            check=False,
        ).returncode
        == 0
    )
    if not tracked_source_clean:
        raise RuntimeError("current production source paths have uncommitted changes")

    print("[stage] importing exact current and remote implementations", flush=True)
    import flash_attn
    import torch
    import torch.nn.functional as F
    import transformers

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("this reference probe requires CUDA")
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    remote_duo = load_git_module(
        repo_root,
        REMOTE_COMMIT,
        "semicat/net/duo.py",
        "m2_probe_remote_duo",
    )
    remote_sampling = load_git_module(
        repo_root,
        REMOTE_COMMIT,
        "block/sampling.py",
        "m2_probe_remote_sampling",
    )

    from block.block_dit import BlockDIT
    from block.sampling import block_causal_sample
    import semicat.net.duo as current_duo

    # The immutable remote sampler imports these two helpers by their production
    # package name inside _blockwise_sample_cached. Patch only this process so the
    # unmodified remote source can execute against the dynamically loaded remote net.
    current_duo.empty_kv_cache = remote_duo.empty_kv_cache
    current_duo.cache_seq_len = remote_duo.cache_seq_len

    print("[stage] loading checkpoint and strict-loading both networks", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint)
    net_state = normalize_net_state(raw_state)
    checkpoint_global_step = checkpoint.get("global_step")

    remote_net = remote_duo.DIT(**MODEL_CONFIG)
    current_net = BlockDIT(
        **MODEL_CONFIG,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        jvp_attention_backend="math",
        flex_kernel_block_size=64,
    )
    remote_load = remote_net.load_state_dict(net_state, strict=True)
    current_load = current_net.load_state_dict(net_state, strict=True)
    namespace_equal = list(remote_net.state_dict()) == list(current_net.state_dict())
    parameter_dtypes = sorted({str(value.dtype) for value in net_state.values()})
    del checkpoint, raw_state, net_state

    remote_net = remote_net.to(device).eval()
    current_net = current_net.to(device).eval()
    remote_module = SamplerModule(torch, remote_net)
    current_module = SamplerModule(torch, current_net)

    print("[stage] paired single-block logit/probability probes", flush=True)
    length = MODEL_CONFIG["length"]
    vocab_size = MODEL_CONFIG["vocab_size"]
    reset_rng(torch, LOGIT_PROBE_SEED)
    clean_tokens = torch.randint(0, vocab_size, (1, length), device=device)
    clean = F.one_hot(clean_tokens, vocab_size).to(torch.float32)
    kv_cache = remote_duo.empty_kv_cache(len(remote_net.blocks))
    logit_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for block_index in range(length // BLOCK_SIZE):
            lo = block_index * BLOCK_SIZE
            hi = lo + BLOCK_SIZE
            positions = torch.arange(lo, hi, device=device)
            z_block = torch.randn(
                1, BLOCK_SIZE, vocab_size, device=device, dtype=torch.float32
            )

            if block_index in LOGIT_PROBE_BLOCKS:
                s_scalar = torch.full(
                    (1,), LOGIT_PROBE_S, device=device, dtype=torch.float32
                )
                t_scalar = torch.full(
                    (1,), LOGIT_PROBE_T, device=device, dtype=torch.float32
                )
                cached_logits = remote_net.forward_block(
                    z_block, s_scalar, t_scalar, positions, kv_cache
                )

                noisy = torch.zeros(
                    1, length, vocab_size, device=device, dtype=torch.float32
                )
                noisy[:, lo:hi] = z_block
                s_tokens = torch.zeros(1, length, device=device)
                t_tokens = torch.zeros(1, length, device=device)
                s_tokens[:, lo:hi] = LOGIT_PROBE_S
                t_tokens[:, lo:hi] = LOGIT_PROBE_T
                ones = torch.ones(1, length, device=device)
                doubled_input = torch.cat((clean, noisy), dim=1)
                doubled_s = torch.cat((ones, s_tokens), dim=1)
                doubled_t = torch.cat((ones, t_tokens), dim=1)
                full_logits = current_net(
                    doubled_input, doubled_s, doubled_t
                )[:, length + lo : length + hi]

                logit_abs = (cached_logits - full_logits).abs()
                cached_probabilities = cached_logits.softmax(dim=-1)
                full_probabilities = full_logits.softmax(dim=-1)
                probability_abs = (
                    cached_probabilities - full_probabilities
                ).abs()
                row = {
                    "block_index": block_index,
                    "positions": [lo, hi - 1],
                    "remote_cache_length": int(
                        remote_duo.cache_seq_len(kv_cache)
                    ),
                    "logits_max_abs_diff": float(logit_abs.max()),
                    "logits_mean_abs_diff": float(logit_abs.mean()),
                    "probabilities_max_abs_diff": float(probability_abs.max()),
                    "probabilities_mean_abs_diff": float(probability_abs.mean()),
                    "argmax_disagreements": int(
                        (
                            cached_logits.argmax(dim=-1)
                            != full_logits.argmax(dim=-1)
                        ).sum()
                    ),
                    "argmax_total": int(cached_logits.shape[0] * cached_logits.shape[1]),
                }
                logit_rows.append(row)
                print(f"[logit] {json.dumps(row, sort_keys=True)}", flush=True)
                del (
                    cached_logits,
                    full_logits,
                    noisy,
                    doubled_input,
                    doubled_s,
                    doubled_t,
                    logit_abs,
                    cached_probabilities,
                    full_probabilities,
                    probability_abs,
                )

            if block_index + 1 < length // BLOCK_SIZE:
                kv_cache = remote_net.encode_clean(
                    clean[:, lo:hi], positions, kv_cache
                )

    del clean_tokens, clean, kv_cache
    torch.cuda.empty_cache()

    print("[stage] paired end-to-end token probes", flush=True)
    # This matches the actual current dump path: compiled FlexAttention over 2L.
    current_net.attention_backend = "flex"
    current_net.jvp_attention_backend = "auto"
    e2e_rows: list[dict[str, Any]] = []

    for discretize in E2E_DISCRETIZE:
        for nfe in E2E_NFE:
            seed = E2E_SEED_BASE + nfe
            reset_rng(torch, seed)
            current_tokens = block_causal_sample(
                current_module,
                BLOCK_SIZE,
                nfe,
                batch_size=E2E_BATCH_SIZE,
                length=length,
                discretize=discretize,
            )
            reset_rng(torch, seed)
            remote_tokens = remote_sampling.blockwise_sample(
                remote_module,
                BLOCK_SIZE,
                nfe,
                batch_size=E2E_BATCH_SIZE,
                length=length,
                discretize=discretize,
            )
            torch.cuda.synchronize(device)
            mismatches = int((current_tokens != remote_tokens).sum())
            total = int(current_tokens.numel())
            row = {
                "nfe_per_block": nfe,
                "discretize": discretize,
                "seed": seed,
                "batch_size": E2E_BATCH_SIZE,
                "length": length,
                "mismatch_tokens": mismatches,
                "total_tokens": total,
                "mismatch_rate": mismatches / total,
                "exact_match": bool(mismatches == 0),
                "flow_forwards_per_sequence_each_arm": (length // BLOCK_SIZE) * nfe,
                "remote_cache_encode_forwards_per_sequence": (length // BLOCK_SIZE) - 1,
            }
            e2e_rows.append(row)
            print(f"[e2e] {json.dumps(row, sort_keys=True)}", flush=True)

    properties = torch.cuda.get_device_properties(device)
    total_e2e_mismatches = sum(row["mismatch_tokens"] for row in e2e_rows)
    total_e2e_tokens = sum(row["total_tokens"] for row in e2e_rows)
    source_records = {
        "current": [
            source_record(repo_root, CURRENT_COMMIT, path)
            for path in CURRENT_SOURCE_PATHS
        ],
        "remote": [
            source_record(repo_root, REMOTE_COMMIT, path)
            for path in REMOTE_SOURCE_PATHS
        ],
    }

    result = {
        "schema": "bcfm-m2-implementation-equivalence/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Same TinyStories M1 state: current full masked-2L M2 versus "
            "remote clean-prefix KV-cached M2"
        ),
        "generated_by": str(Path(__file__).resolve().relative_to(repo_root)),
        "generated_by_sha256": sha256_file(Path(__file__).resolve()),
        "repository": str(repo_root),
        "revisions": {
            "current_expected_commit": CURRENT_COMMIT,
            "current_actual_head": current_head,
            "current_branch": str(run_git(repo_root, "branch", "--show-current")),
            "current_source_paths_clean": tracked_source_clean,
            "remote_commit": REMOTE_COMMIT,
            "remote_branch_alias": REMOTE_BRANCH_ALIAS,
            "remote_branch_alias_target_at_run": remote_alias_target,
            "merge_base": str(
                run_git(repo_root, "merge-base", CURRENT_COMMIT, REMOTE_COMMIT)
            ),
        },
        "sources": source_records,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "sha256_matches_expected": checkpoint_sha256
            == EXPECTED_CHECKPOINT_SHA256,
            "global_step": checkpoint_global_step,
            "network_state_tensors": len(remote_net.state_dict()),
            "network_state_dtypes": parameter_dtypes,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "flash_attn": getattr(flash_attn, "__version__", "unknown"),
            "transformers": transformers.__version__,
            "device_argument": args.device,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensor_dtype": "torch.float32",
            "autocast_enabled": False,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "model": {
            **MODEL_CONFIG,
            "block_size": BLOCK_SIZE,
            "prior_type": "gaussian",
            "eval_mode": True,
            "current_state_namespace_equals_remote": namespace_equal,
            "current_strict_load_missing_keys": list(current_load.missing_keys),
            "current_strict_load_unexpected_keys": list(
                current_load.unexpected_keys
            ),
            "remote_strict_load_missing_keys": list(remote_load.missing_keys),
            "remote_strict_load_unexpected_keys": list(remote_load.unexpected_keys),
        },
        "logit_probe": {
            "purpose": "isolate sampler semantics with SDPA on both arms",
            "current_execution": "BlockDIT full masked [clean; noisy] 2L",
            "current_attention_backend": "sdpa",
            "remote_execution": "DuoDIT.forward_block against clean-prefix KV cache",
            "remote_attention_backend": "sdpa",
            "batch_size": 1,
            "seed": LOGIT_PROBE_SEED,
            "length": length,
            "block_size": BLOCK_SIZE,
            "s": LOGIT_PROBE_S,
            "t": LOGIT_PROBE_T,
            "blocks": list(LOGIT_PROBE_BLOCKS),
            "clean_tokens": "torch.randint after seed reset",
            "noisy_blocks": "sequential torch.randn draw for every block",
            "results": logit_rows,
            "aggregate": {
                "max_logits_abs_diff": max(
                    row["logits_max_abs_diff"] for row in logit_rows
                ),
                "argmax_disagreements": sum(
                    row["argmax_disagreements"] for row in logit_rows
                ),
                "argmax_total": sum(row["argmax_total"] for row in logit_rows),
            },
        },
        "e2e_probe": {
            "purpose": "compare final tokens through the two actual execution backends",
            "current_execution": "block_causal_sample + BlockDIT full masked 2L",
            "current_attention_backend": "compiled FlexAttention",
            "remote_execution": "immutable remote blockwise_sample + DuoDIT KV cache",
            "remote_attention_backend": "SDPA",
            "length": length,
            "block_size": BLOCK_SIZE,
            "batch_size": E2E_BATCH_SIZE,
            "nfe_per_block": list(E2E_NFE),
            "discretize": list(E2E_DISCRETIZE),
            "seed_rule": "9000 + nfe_per_block, reset identically before each arm",
            "results": e2e_rows,
            "aggregate": {
                "mismatch_tokens": total_e2e_mismatches,
                "total_tokens": total_e2e_tokens,
                "mismatch_rate": total_e2e_mismatches / total_e2e_tokens,
                "all_exact_match": bool(total_e2e_mismatches == 0),
            },
        },
        "conclusion": {
            "supported": bool(
                namespace_equal
                and not current_load.missing_keys
                and not current_load.unexpected_keys
                and not remote_load.missing_keys
                and not remote_load.unexpected_keys
                and sum(row["argmax_disagreements"] for row in logit_rows) == 0
                and total_e2e_mismatches == 0
            ),
            "statement": (
                "Within the recorded fp32 probes, remote cached M2 is numerically "
                "consistent with the current masked-2L M2 execution of the same state."
            ),
        },
        "caveats": [
            "This finite probe is not a proof for every input, NFE, batch size, or dtype.",
            "The end-to-end arms intentionally use different attention backends "
            "(current FlexAttention, remote SDPA), so tiny floating-point differences are expected.",
            f"Float32 matmul precision was explicitly pinned to "
            f"{args.matmul_precision!r}. SemicatModule.__init__ selects 'high' "
            "(TF32-capable), so production comparisons should use the corresponding artifact.",
            "The end-to-end probe covers NFE 1/2 and batch size 2 only; a fair quality "
            "experiment still needs 512 paired samples over the full requested NFE grid.",
            "No latency conclusion is drawn: compiler warmup, throughput, and synchronized "
            "batch-1 latency require a separate benchmark protocol.",
            "The remote sampler's internal import is satisfied by process-local helper "
            "bindings; no repository module or production file is modified.",
        ],
    }

    print("[stage] writing evidence artifact", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[done] wrote={output_path}", flush=True)
    print(
        "[done] summary="
        + json.dumps(
            {
                "max_logits_abs_diff": result["logit_probe"]["aggregate"][
                    "max_logits_abs_diff"
                ],
                "logit_argmax_disagreements": result["logit_probe"][
                    "aggregate"
                ]["argmax_disagreements"],
                "e2e_mismatch_tokens": total_e2e_mismatches,
                "e2e_total_tokens": total_e2e_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
