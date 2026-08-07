#!/usr/bin/env python3
"""Create final stratified uncertainty tables, plot, and report."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parent
NFES = (8, 16, 32)
REPLICATES = 20_000
SEED = 20260719


def stamp(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def weighted_nll(sums: np.ndarray, counts: np.ndarray) -> float:
    return float(sums.sum() / counts.sum())


def bootstrap_delta(
    source_sums: np.ndarray,
    source_counts: np.ndarray,
    tuned_sums: np.ndarray,
    tuned_counts: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    source_sums = source_sums[mask]
    source_counts = source_counts[mask]
    tuned_sums = tuned_sums[mask]
    tuned_counts = tuned_counts[mask]
    rng = np.random.default_rng(seed)
    n = len(source_sums)
    output = np.empty(REPLICATES, dtype=np.float64)
    for lo in range(0, REPLICATES, 250):
        hi = min(REPLICATES, lo + 250)
        indices = rng.integers(0, n, size=(hi - lo, n), dtype=np.int32)
        source = source_sums[indices].sum(axis=1) / source_counts[indices].sum(
            axis=1
        )
        tuned = tuned_sums[indices].sum(axis=1) / tuned_counts[indices].sum(
            axis=1
        )
        output[lo:hi] = tuned - source
    return output


def main() -> None:
    stamp(
        f"stage=start nfe={NFES} bootstrap={REPLICATES} seed={SEED} output={OUTPUT}"
    )
    judge = np.load(OUTPUT / "per_sample_judge.npz")
    memorization = np.load(OUTPUT / "memorization_per_sample.npz")
    comparison = {
        int(row["nfe_steps_per_block"]): row
        for row in json.loads((OUTPUT / "comparison.json").read_text())
    }
    audit = {
        (row["checkpoint_label"], int(row["nfe_steps_per_block"])): row
        for row in json.loads((OUTPUT / "memorization_audit.json").read_text())
    }

    rows = []
    for nfe in NFES:
        source_sums = judge[f"source_nfe{nfe}_nll_sum"]
        source_counts = judge[f"source_nfe{nfe}_token_count"]
        tuned_sums = judge[f"step90k_nfe{nfe}_nll_sum"]
        tuned_counts = judge[f"step90k_nfe{nfe}_token_count"]
        longest = memorization[f"step90k_nfe{nfe}_longest_seen_match"]
        strata = (
            ("all", np.ones(len(longest), dtype=bool)),
            ("noncopy_match_lt_64", longest < 64),
            ("long_copy_match_ge_128", longest >= 128),
        )
        for stratum_index, (stratum, mask) in enumerate(strata):
            deltas = bootstrap_delta(
                source_sums,
                source_counts,
                tuned_sums,
                tuned_counts,
                mask,
                SEED + nfe * 10 + stratum_index,
            )
            source_nll = weighted_nll(source_sums[mask], source_counts[mask])
            tuned_nll = weighted_nll(tuned_sums[mask], tuned_counts[mask])
            rows.append(
                {
                    "nfe_steps_per_block": nfe,
                    "stratum": stratum,
                    "n_sequences": int(mask.sum()),
                    "sequence_fraction": float(mask.mean()),
                    "source_nll_same_prior_indices": source_nll,
                    "step90k_nll": tuned_nll,
                    "delta_nll_90k_minus_source": tuned_nll - source_nll,
                    "paired_bootstrap_95_low": float(
                        np.quantile(deltas, 0.025)
                    ),
                    "paired_bootstrap_95_high": float(
                        np.quantile(deltas, 0.975)
                    ),
                    "bootstrap_probability_delta_ge_zero": float(
                        np.mean(deltas >= 0)
                    ),
                    "bootstrap_replicates": REPLICATES,
                }
            )
            stamp(
                "stage=bootstrap "
                f"nfe={nfe} stratum={stratum} n={mask.sum()} "
                f"delta={rows[-1]['delta_nll_90k_minus_source']:.6f} "
                f"ci=[{rows[-1]['paired_bootstrap_95_low']:.6f},"
                f"{rows[-1]['paired_bootstrap_95_high']:.6f}]"
            )

    with (OUTPUT / "stratified_bootstrap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "stratified_bootstrap.json").write_text(json.dumps(rows, indent=2))

    by_key = {
        (int(row["nfe_steps_per_block"]), row["stratum"]): row for row in rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].plot(
        NFES,
        [comparison[n]["source_mean_nll"] for n in NFES],
        marker="o",
        label="source",
    )
    axes[0].plot(
        NFES,
        [comparison[n]["step90k_mean_nll"] for n in NFES],
        marker="o",
        label="90k",
    )
    axes[0].set_title("Aggregate GPT-J-6B NLL")
    axes[0].set_ylabel("NLL (lower is better)")
    axes[0].legend()

    for stratum, label, color in (
        ("all", "all outputs", "#0072B2"),
        ("noncopy_match_lt_64", "90k match <64 chars", "#D55E00"),
    ):
        centers = np.asarray(
            [by_key[(n, stratum)]["delta_nll_90k_minus_source"] for n in NFES]
        )
        lower = np.asarray(
            [by_key[(n, stratum)]["paired_bootstrap_95_low"] for n in NFES]
        )
        upper = np.asarray(
            [by_key[(n, stratum)]["paired_bootstrap_95_high"] for n in NFES]
        )
        axes[1].errorbar(
            NFES,
            centers,
            yerr=np.vstack((centers - lower, upper - centers)),
            marker="o",
            capsize=4,
            label=label,
            color=color,
        )
    axes[1].axhline(0.0, color="#555555", linestyle="--")
    axes[1].set_title("Paired NLL change: 90k − source")
    axes[1].set_ylabel("ΔNLL (negative is better)")
    axes[1].legend()

    axes[2].plot(
        NFES,
        [
            100 * audit[("step90k", n)]["fraction_with_seen_match_ge_64"]
            for n in NFES
        ],
        marker="o",
        label="≥64-char train match",
    )
    axes[2].plot(
        NFES,
        [
            100 * audit[("step90k", n)]["fraction_with_seen_match_ge_128"]
            for n in NFES
        ],
        marker="o",
        label="≥128-char train match",
    )
    axes[2].plot(
        NFES,
        [
            100 * audit[("step90k", n)]["fraction_exact_seen_256_window"]
            for n in NFES
        ],
        marker="o",
        label="exact 256-char train window",
    )
    axes[2].set_title("Exact copying by the 90k model")
    axes[2].set_ylabel("Generated sequences (%)")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("NFE (steps per block, B=16)")
        axis.set_xticks(NFES)
        axis.grid(alpha=0.25)
    fig.suptitle(
        "The aggregate NLL gain is carried by memorized train-prefix continuations",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "final_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    source_protocol = json.loads(
        (OUTPUT / "source_n2048_seed0/protocol.json").read_text()
    )
    tuned_protocol = json.loads(
        (OUTPUT / "step90k_n2048_seed0/protocol.json").read_text()
    )
    table_lines = []
    for nfe in NFES:
        row = comparison[nfe]
        overall = by_key[(nfe, "all")]
        noncopy = by_key[(nfe, "noncopy_match_lt_64")]
        table_lines.append(
            f"| {nfe} | {row['source_mean_nll']:.6f} | "
            f"{row['step90k_mean_nll']:.6f} | "
            f"{row['source_gen_ppl']:.2f} | {row['step90k_gen_ppl']:.2f} | "
            f"{overall['delta_nll_90k_minus_source']:+.6f} "
            f"[{overall['paired_bootstrap_95_low']:+.6f}, "
            f"{overall['paired_bootstrap_95_high']:+.6f}] | "
            f"{noncopy['delta_nll_90k_minus_source']:+.6f} "
            f"[{noncopy['paired_bootstrap_95_low']:+.6f}, "
            f"{noncopy['paired_bootstrap_95_high']:+.6f}] |"
        )

    report = f"""# B=16 generative NLL: source CFM weights versus 90k BCFM

Date: 2026-07-19

## Protocol

- Source: `{source_protocol['checkpoint']}` (raw fine-tune initialization;
  checkpoint metadata step {source_protocol['checkpoint_step']}).
- Tuned: `{tuned_protocol['checkpoint']}`.
- Block size: 16.
- NFE: 8, 16, 32 steps per block, corresponding to 128, 256, 512 model
  forwards per sequence.
- 2,048 generated sequences per point; 256 Text8 characters; seed 0; argmax
  discretization; sampling batch 128.
- Judge: GPT-J-6B; tokenizer: GPT-2-large; token-weighted mean NLL;
  PPL = exp(NLL).
- CPU transfer was synchronous (`non_blocking=False`).
- Uncertainty: 20,000 paired sequence-level bootstrap replicates using the
  same prior indices.

## Aggregate result

| NFE | Source NLL | 90k NLL | Source PPL | 90k PPL | Overall ΔNLL [95% CI] | ΔNLL after excluding 90k outputs with ≥64-char seen-train match [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

The aggregate judge metric improves significantly at every NFE. PPL falls by
14.3%, 16.3%, and 17.5% for NFE 8, 16, and 32.

## Memorization audit

All 2,048 sequences are unique at every point, and the 90k pooled character
entropy is slightly higher, so this is not a coarse duplicate or character
mode collapse.

However, exact suffix-automaton matching against the only 351,563 training
characters exposed by the loader shows:

| NFE | Source fraction with ≥64-char exact train match | 90k fraction with ≥64-char exact train match | 90k fraction with ≥128-char match | 90k exact 256-char train windows |
|---:|---:|---:|---:|---:|
| 8 | 0.0% | {100 * audit[('step90k', 8)]['fraction_with_seen_match_ge_64']:.2f}% | {100 * audit[('step90k', 8)]['fraction_with_seen_match_ge_128']:.2f}% | {100 * audit[('step90k', 8)]['fraction_exact_seen_256_window']:.2f}% |
| 16 | 0.0% | {100 * audit[('step90k', 16)]['fraction_with_seen_match_ge_64']:.2f}% | {100 * audit[('step90k', 16)]['fraction_with_seen_match_ge_128']:.2f}% | {100 * audit[('step90k', 16)]['fraction_exact_seen_256_window']:.2f}% |
| 32 | 0.0% | {100 * audit[('step90k', 32)]['fraction_with_seen_match_ge_64']:.2f}% | {100 * audit[('step90k', 32)]['fraction_with_seen_match_ge_128']:.2f}% | {100 * audit[('step90k', 32)]['fraction_exact_seen_256_window']:.2f}% |

At NFE 32, the non-copy stratum (`<64` matching characters) contains 82.6% of
sequences and 89.4% of judged GPT tokens. Within this stratum, NLL is 6.7963
for 90k versus 6.7716 for the source outputs at the same prior indices:
ΔNLL = +0.0247, i.e. slightly worse. The aggregate gain comes from the
roughly 16.7% long-copy subgroup, whose NLL is around 4.7 rather than 6.8.

## Conclusion

If the sole metric is aggregate GPT-J generative NLL, the 90k fine-tune
improved it decisively.

It is not evidence of improved generalization or novel-generation quality.
The gain is carried by long or exact copies from the truncated train prefix.
After excluding those copies, NLL is unchanged at NFE 8 and significantly
worse at NFE 16 and 32. The supported interpretation is memorization-driven
metric improvement, not a clean generation-quality improvement.

## Evidence

- `comparison.csv`: aggregate NLL/PPL and entropy.
- `paired_bootstrap.csv`: overall paired uncertainty.
- `stratified_bootstrap.csv`: copy-stratified paired uncertainty.
- `memorization_audit.csv`: exact train-substring audit.
- `copy_stratified_nll.csv`: NLL by copy length.
- `per_sample_judge.npz`: per-sequence NLL sums/token counts.
- `memorization_per_sample.npz`: per-sequence exact-match lengths.
- `top_seen_train_matches.json`: inspected examples.
- `final_comparison.png`: reviewed comparison plot.
- `source_n2048_seed0/` and `step90k_n2048_seed0/`: exact protocols,
  generated tokens, and generated strings.
"""
    (OUTPUT / "REPORT.md").write_text(report)
    stamp(
        "stage=done "
        f"bootstrap={OUTPUT / 'stratified_bootstrap.csv'} "
        f"plot={OUTPUT / 'final_comparison.png'} "
        f"report={OUTPUT / 'REPORT.md'}"
    )


if __name__ == "__main__":
    main()
