#!/usr/bin/env python3
"""Per-sequence GPT-J NLL, paired bootstrap, and copy-stratified comparison."""

from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn import functional as F


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT))

from semicat.metric.text_dist import TextMetrics  # noqa: E402


RUNS = {
    "source": OUTPUT / "source_n2048_seed0",
    "step90k": OUTPUT / "step90k_n2048_seed0",
}
NFES = (8, 16, 32)
JUDGE_MODEL = "EleutherAI/gpt-j-6B"
TOKENIZER_MODEL = "gpt2-large"
JUDGE_BATCH_SIZE = 8
CONTEXT_SIZE = 256
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260719


def stamp(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def strings_from_tokens(tokens: torch.Tensor, itos: dict[int, str]) -> list[str]:
    return [
        "".join(itos[int(token)] for token in sequence)
        for sequence in tokens.tolist()
    ]


@torch.inference_mode()
def evaluate(
    strings: list[str],
    tokenizer,
    judge,
    *,
    label: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    samples, attention_mask = TextMetrics._retokenize(
        tokenizer,
        CONTEXT_SIZE,
        strings,
        device,
    )
    if samples.size(0) % JUDGE_BATCH_SIZE:
        raise ValueError("sample count must be divisible by judge batch size")
    nll_sums = np.zeros(samples.size(0), dtype=np.float64)
    token_counts = np.zeros(samples.size(0), dtype=np.int64)
    n_batches = samples.size(0) // JUDGE_BATCH_SIZE
    started = time.perf_counter()
    for batch_index in range(n_batches):
        lo = batch_index * JUDGE_BATCH_SIZE
        hi = lo + JUDGE_BATCH_SIZE
        sample_batch = samples[lo:hi]
        mask_batch = attention_mask[lo:hi]
        logits = judge(sample_batch, attention_mask=mask_batch).logits
        nll = F.cross_entropy(
            logits.transpose(-1, -2)[..., :-1],
            sample_batch[..., 1:],
            reduction="none",
        )
        first_eos = (sample_batch == tokenizer.eos_token_id).cumsum(-1) == 1
        token_mask = sample_batch != tokenizer.eos_token_id
        valid = first_eos[..., 1:] + token_mask[..., 1:]
        nll_sums[lo:hi] = (nll * valid).sum(dim=1).double().cpu().numpy()
        token_counts[lo:hi] = valid.sum(dim=1).cpu().numpy()
        if (
            batch_index == 0
            or (batch_index + 1) % 64 == 0
            or batch_index + 1 == n_batches
        ):
            elapsed = time.perf_counter() - started
            stamp(
                "stage=judge "
                f"label={label} batch={batch_index + 1}/{n_batches} "
                f"elapsed={elapsed:.1f}s batches_per_s="
                f"{(batch_index + 1) / max(elapsed, 1e-9):.2f}"
            )
    return nll_sums, token_counts


def token_weighted_nll(nll_sums: np.ndarray, token_counts: np.ndarray) -> float:
    return float(nll_sums.sum() / token_counts.sum())


def paired_bootstrap(
    source_sums: np.ndarray,
    source_counts: np.ndarray,
    tuned_sums: np.ndarray,
    tuned_counts: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(source_sums)
    deltas = np.empty(replicates, dtype=np.float64)
    chunk_size = 250
    for lo in range(0, replicates, chunk_size):
        hi = min(replicates, lo + chunk_size)
        indices = rng.integers(0, n, size=(hi - lo, n), dtype=np.int32)
        source_nll = source_sums[indices].sum(axis=1) / source_counts[indices].sum(
            axis=1
        )
        tuned_nll = tuned_sums[indices].sum(axis=1) / tuned_counts[indices].sum(
            axis=1
        )
        deltas[lo:hi] = tuned_nll - source_nll
    return deltas


def main() -> None:
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Match SemicatModule.__init__ and therefore the original comparison run.
    torch.set_float32_matmul_precision("high")
    stamp(
        "stage=start "
        f"device={device} gpu={torch.cuda.get_device_name(0)} "
        f"judge={JUDGE_MODEL} tokenizer={TOKENIZER_MODEL} "
        f"dtype=float32 matmul_precision={torch.get_float32_matmul_precision()} "
        f"batch={JUDGE_BATCH_SIZE} seed={BOOTSTRAP_SEED} "
        f"bootstrap={BOOTSTRAP_REPLICATES} output={OUTPUT}"
    )
    with (ROOT / "data/text8/meta.pkl").open("rb") as handle:
        itos = pickle.load(handle)["itos"]
    memorization = np.load(OUTPUT / "memorization_per_sample.npz")
    expected_summaries = {
        label: {
            int(row["nfe"]): row
            for row in json.loads((directory / "summary.json").read_text())
        }
        for label, directory in RUNS.items()
    }

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    stamp(f"stage=judge_load model={JUDGE_MODEL} tokenizer={TOKENIZER_MODEL}")
    tokenizer = TextMetrics._load_tokenizer(TOKENIZER_MODEL)
    judge = transformers.AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL
    ).eval().to(device)
    stamp(
        "stage=judge_ready "
        f"dtype={next(judge.parameters()).dtype} parameters="
        f"{sum(parameter.numel() for parameter in judge.parameters())}"
    )

    arrays: dict[str, np.ndarray] = {}
    aggregate_rows: list[dict] = []
    for label, directory in RUNS.items():
        for nfe in NFES:
            key = f"{label}_nfe{nfe}"
            token_path = directory / f"blockwise_cfm_nfe{nfe}_tokens.pt"
            tokens = torch.load(token_path, map_location="cpu", weights_only=True)
            strings = strings_from_tokens(tokens, itos)
            sums, counts = evaluate(
                strings,
                tokenizer,
                judge,
                label=key,
                device=device,
            )
            arrays[f"{key}_nll_sum"] = sums
            arrays[f"{key}_token_count"] = counts
            measured = token_weighted_nll(sums, counts)
            expected = float(expected_summaries[label][nfe]["mean_nll"])
            difference = measured - expected
            # The original evaluator reduces each batch to one float32 scalar,
            # whereas this audit retains per-sequence sums. The resulting
            # floating-point summation-order difference is around 1e-4.
            if abs(difference) > 1e-3:
                raise RuntimeError(
                    f"per-sample reduction mismatch for {key}: "
                    f"measured={measured}, expected={expected}"
                )
            aggregate_rows.append(
                {
                    "checkpoint_label": label,
                    "nfe_steps_per_block": nfe,
                    "mean_nll": measured,
                    "gen_ppl": float(np.exp(measured)),
                    "effective_judge_tokens": int(counts.sum()),
                    "expected_summary_nll": expected,
                    "reduction_difference": difference,
                    "mean_per_sequence_nll": float(np.mean(sums / counts)),
                    "std_per_sequence_nll": float(np.std(sums / counts, ddof=1)),
                }
            )
            np.savez_compressed(OUTPUT / "per_sample_judge.npz", **arrays)
            stamp(
                "stage=judge_set_done "
                f"label={key} mean_nll={measured:.6f} "
                f"ppl={np.exp(measured):.3f} effective_tokens={counts.sum()}"
            )

    with (OUTPUT / "per_sample_judge_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (OUTPUT / "per_sample_judge_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2)
    )

    bootstrap_rows: list[dict] = []
    strata_rows: list[dict] = []
    for nfe in NFES:
        source_sums = arrays[f"source_nfe{nfe}_nll_sum"]
        source_counts = arrays[f"source_nfe{nfe}_token_count"]
        tuned_sums = arrays[f"step90k_nfe{nfe}_nll_sum"]
        tuned_counts = arrays[f"step90k_nfe{nfe}_token_count"]
        deltas = paired_bootstrap(
            source_sums,
            source_counts,
            tuned_sums,
            tuned_counts,
            seed=BOOTSTRAP_SEED + nfe,
            replicates=BOOTSTRAP_REPLICATES,
        )
        point_delta = token_weighted_nll(
            tuned_sums, tuned_counts
        ) - token_weighted_nll(source_sums, source_counts)
        bootstrap_rows.append(
            {
                "nfe_steps_per_block": nfe,
                "delta_nll_90k_minus_source": point_delta,
                "paired_bootstrap_mean": float(deltas.mean()),
                "paired_bootstrap_std": float(deltas.std(ddof=1)),
                "paired_bootstrap_95_low": float(np.quantile(deltas, 0.025)),
                "paired_bootstrap_95_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_probability_delta_ge_zero": float(np.mean(deltas >= 0)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED + nfe,
            }
        )

        longest = memorization[f"step90k_nfe{nfe}_longest_seen_match"]
        strata = (
            ("all", np.ones(len(longest), dtype=bool)),
            ("match_lt_32", longest < 32),
            ("match_lt_64", longest < 64),
            ("match_32_63", (longest >= 32) & (longest < 64)),
            ("match_64_127", (longest >= 64) & (longest < 128)),
            ("match_128_255", (longest >= 128) & (longest < 256)),
            ("exact_seen_256", longest >= 256),
        )
        for stratum, mask in strata:
            if not mask.any():
                continue
            source_nll = token_weighted_nll(
                source_sums[mask], source_counts[mask]
            )
            tuned_nll = token_weighted_nll(tuned_sums[mask], tuned_counts[mask])
            strata_rows.append(
                {
                    "nfe_steps_per_block": nfe,
                    "stratum_defined_by_90k_longest_seen_match": stratum,
                    "n_sequences": int(mask.sum()),
                    "sequence_fraction": float(mask.mean()),
                    "source_nll_same_prior_indices": source_nll,
                    "step90k_nll": tuned_nll,
                    "delta_nll_90k_minus_source": tuned_nll - source_nll,
                    "step90k_effective_judge_tokens": int(tuned_counts[mask].sum()),
                    "step90k_fraction_of_judge_tokens": float(
                        tuned_counts[mask].sum() / tuned_counts.sum()
                    ),
                }
            )
        stamp(
            "stage=paired_analysis "
            f"nfe={nfe} delta={point_delta:.6f} "
            f"ci95=[{bootstrap_rows[-1]['paired_bootstrap_95_low']:.6f},"
            f"{bootstrap_rows[-1]['paired_bootstrap_95_high']:.6f}]"
        )

    with (OUTPUT / "paired_bootstrap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader()
        writer.writerows(bootstrap_rows)
    (OUTPUT / "paired_bootstrap.json").write_text(
        json.dumps(bootstrap_rows, indent=2)
    )
    with (OUTPUT / "copy_stratified_nll.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(strata_rows[0]))
        writer.writeheader()
        writer.writerows(strata_rows)
    (OUTPUT / "copy_stratified_nll.json").write_text(
        json.dumps(strata_rows, indent=2)
    )
    stamp(
        "stage=done "
        f"per_sample={OUTPUT / 'per_sample_judge.npz'} "
        f"bootstrap={OUTPUT / 'paired_bootstrap.csv'} "
        f"strata={OUTPUT / 'copy_stratified_nll.csv'}"
    )


if __name__ == "__main__":
    main()
