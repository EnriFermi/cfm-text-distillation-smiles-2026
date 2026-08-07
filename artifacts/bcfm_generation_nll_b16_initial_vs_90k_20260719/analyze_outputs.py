#!/usr/bin/env python3
"""Compare B=16 source/90k generations and audit memorization/diversity."""

from __future__ import annotations

import csv
import json
import math
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
RUNS = {
    "source": OUTPUT / "source_n2048_seed0",
    "step90k": OUTPUT / "step90k_n2048_seed0",
}
NFES = (8, 16, 32)
BLOCK_SIZE = 16
TRAIN_PREFIX_TOKENS = 351_563


def stamp(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_suffix_automaton(sequence: np.ndarray, vocab_size: int) -> dict[str, np.ndarray | int]:
    """Dense-alphabet suffix automaton for exact longest-substring queries."""
    max_states = 2 * len(sequence)
    transitions = np.full((max_states, vocab_size), -1, dtype=np.int32)
    suffix_link = np.full(max_states, -1, dtype=np.int32)
    state_length = np.zeros(max_states, dtype=np.int32)
    first_position = np.full(max_states, -1, dtype=np.int32)
    size = 1
    last = 0

    for position, raw_symbol in enumerate(sequence):
        symbol = int(raw_symbol)
        current = size
        size += 1
        state_length[current] = state_length[last] + 1
        first_position[current] = position
        previous = last
        while previous >= 0 and transitions[previous, symbol] == -1:
            transitions[previous, symbol] = current
            previous = int(suffix_link[previous])
        if previous < 0:
            suffix_link[current] = 0
        else:
            target = int(transitions[previous, symbol])
            if state_length[previous] + 1 == state_length[target]:
                suffix_link[current] = target
            else:
                clone = size
                size += 1
                transitions[clone] = transitions[target]
                state_length[clone] = state_length[previous] + 1
                suffix_link[clone] = suffix_link[target]
                first_position[clone] = first_position[target]
                while previous >= 0 and transitions[previous, symbol] == target:
                    transitions[previous, symbol] = clone
                    previous = int(suffix_link[previous])
                suffix_link[target] = clone
                suffix_link[current] = clone
        last = current

        if (position + 1) % 50_000 == 0 or position + 1 == len(sequence):
            stamp(
                "stage=suffix_automaton "
                f"tokens={position + 1}/{len(sequence)} states={size}"
            )

    return {
        "transitions": transitions[:size],
        "suffix_link": suffix_link[:size],
        "state_length": state_length[:size],
        "first_position": first_position[:size],
        "states": size,
    }


def longest_seen_substring(
    query: np.ndarray,
    automaton: dict[str, np.ndarray | int],
) -> tuple[int, int, int]:
    transitions = automaton["transitions"]
    suffix_link = automaton["suffix_link"]
    state_length = automaton["state_length"]
    first_position = automaton["first_position"]
    assert isinstance(transitions, np.ndarray)
    assert isinstance(suffix_link, np.ndarray)
    assert isinstance(state_length, np.ndarray)
    assert isinstance(first_position, np.ndarray)

    state = 0
    length = 0
    best_length = 0
    best_query_end = -1
    best_source_end = -1
    for query_position, raw_symbol in enumerate(query):
        symbol = int(raw_symbol)
        while state and transitions[state, symbol] == -1:
            state = int(suffix_link[state])
            length = min(length, int(state_length[state]))
        target = int(transitions[state, symbol])
        if target >= 0:
            state = target
            length += 1
        else:
            state = 0
            length = 0
        if length > best_length:
            best_length = length
            best_query_end = query_position
            best_source_end = int(first_position[state])
    return best_length, best_query_end, best_source_end


def decode(tokens: np.ndarray, itos: dict[int, str]) -> str:
    return "".join(itos[int(token)] for token in tokens)


def longest_constant_run(tokens: np.ndarray) -> int:
    if not len(tokens):
        return 0
    changes = np.flatnonzero(tokens[1:] != tokens[:-1]) + 1
    edges = np.concatenate(([0], changes, [len(tokens)]))
    return int(np.diff(edges).max())


def metric_rows() -> list[dict]:
    summaries = {
        label: {
            int(row["nfe"]): row
            for row in json.loads((directory / "summary.json").read_text())
        }
        for label, directory in RUNS.items()
    }
    rows = []
    for nfe in NFES:
        source = summaries["source"][nfe]
        tuned = summaries["step90k"][nfe]
        rows.append(
            {
                "nfe_steps_per_block": nfe,
                "total_forwards_per_sequence": int(tuned["total_model_forwards_per_sequence"]),
                "source_mean_nll": float(source["mean_nll"]),
                "step90k_mean_nll": float(tuned["mean_nll"]),
                "delta_nll_90k_minus_source": float(tuned["mean_nll"] - source["mean_nll"]),
                "source_gen_ppl": float(source["gen_ppl"]),
                "step90k_gen_ppl": float(tuned["gen_ppl"]),
                "delta_ppl_90k_minus_source": float(tuned["gen_ppl"] - source["gen_ppl"]),
                "relative_ppl_change": float(tuned["gen_ppl"] / source["gen_ppl"] - 1.0),
                "source_effective_judge_tokens": int(source["effective_judge_tokens"]),
                "step90k_effective_judge_tokens": int(tuned["effective_judge_tokens"]),
                "source_pooled_char_entropy_nats": float(source["pooled_entropy_nats"]),
                "step90k_pooled_char_entropy_nats": float(tuned["pooled_entropy_nats"]),
                "delta_pooled_char_entropy_nats": float(
                    tuned["pooled_entropy_nats"] - source["pooled_entropy_nats"]
                ),
                "source_mean_per_sample_block_entropy_nats": float(
                    source["mean_per_sample_generation_block_entropy_nats"]
                ),
                "step90k_mean_per_sample_block_entropy_nats": float(
                    tuned["mean_per_sample_generation_block_entropy_nats"]
                ),
            }
        )
    return rows


def main() -> None:
    stamp(
        "stage=start "
        f"runs={RUNS} nfe={NFES} block_size={BLOCK_SIZE} "
        f"train_prefix_tokens={TRAIN_PREFIX_TOKENS} output={OUTPUT}"
    )
    for label, directory in RUNS.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{label}: {directory}")

    metrics = metric_rows()
    with (OUTPUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (OUTPUT / "comparison.json").write_text(json.dumps(metrics, indent=2))

    with (ROOT / "data/text8/meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    itos = meta["itos"]
    train_prefix = np.fromfile(
        ROOT / "data/text8/train.bin",
        dtype=np.uint16,
        count=TRAIN_PREFIX_TOKENS,
    ).astype(np.int16)
    stamp(
        "stage=data_loaded "
        f"train_prefix={ROOT / 'data/text8/train.bin'} tokens={len(train_prefix)} "
        f"vocab={meta['vocab_size']}"
    )
    automaton = build_suffix_automaton(train_prefix, int(meta["vocab_size"]))

    audit_rows: list[dict] = []
    top_matches: dict[str, list[dict]] = {}
    per_sample_arrays: dict[str, np.ndarray] = {}
    for label, directory in RUNS.items():
        for nfe in NFES:
            path = directory / f"blockwise_cfm_nfe{nfe}_tokens.pt"
            tokens = torch.load(path, map_location="cpu", weights_only=True).numpy()
            if tokens.shape != (2048, 256):
                raise ValueError(f"unexpected generated tensor shape {tokens.shape}: {path}")
            stamp(
                "stage=output_audit "
                f"label={label} nfe={nfe} samples={len(tokens)} path={path}"
            )

            unique_sequences = len({row.tobytes() for row in tokens})
            longest_matches = np.zeros(len(tokens), dtype=np.int16)
            aligned_seen_blocks = np.zeros(len(tokens), dtype=np.int16)
            constant_runs = np.zeros(len(tokens), dtype=np.int16)
            examples = []
            for index, row in enumerate(tokens):
                best, query_end, source_end = longest_seen_substring(row, automaton)
                longest_matches[index] = best
                constant_runs[index] = longest_constant_run(row)
                aligned_seen_blocks[index] = sum(
                    longest_seen_substring(
                        row[lo : lo + BLOCK_SIZE], automaton
                    )[0]
                    == BLOCK_SIZE
                    for lo in range(0, row.shape[0], BLOCK_SIZE)
                )
                examples.append(
                    {
                        "sample_index": index,
                        "longest_seen_match": int(best),
                        "generated": decode(row, itos),
                        "generated_match": (
                            decode(row[query_end - best + 1 : query_end + 1], itos)
                            if best
                            else ""
                        ),
                        "seen_train_match": (
                            decode(
                                train_prefix[source_end - best + 1 : source_end + 1],
                                itos,
                            )
                            if best
                            else ""
                        ),
                    }
                )

            key = f"{label}_nfe{nfe}"
            per_sample_arrays[f"{key}_longest_seen_match"] = longest_matches
            per_sample_arrays[f"{key}_aligned_seen_blocks"] = aligned_seen_blocks
            per_sample_arrays[f"{key}_longest_constant_run"] = constant_runs
            top_matches[key] = sorted(
                examples,
                key=lambda item: item["longest_seen_match"],
                reverse=True,
            )[:10]
            audit_rows.append(
                {
                    "checkpoint_label": label,
                    "nfe_steps_per_block": nfe,
                    "n_samples": int(len(tokens)),
                    "unique_sequences": int(unique_sequences),
                    "unique_sequence_fraction": unique_sequences / len(tokens),
                    "mean_longest_seen_train_substring": float(longest_matches.mean()),
                    "median_longest_seen_train_substring": float(
                        np.median(longest_matches)
                    ),
                    "p90_longest_seen_train_substring": float(
                        np.quantile(longest_matches, 0.90)
                    ),
                    "p99_longest_seen_train_substring": float(
                        np.quantile(longest_matches, 0.99)
                    ),
                    "max_longest_seen_train_substring": int(longest_matches.max()),
                    "fraction_with_seen_match_ge_16": float(
                        np.mean(longest_matches >= 16)
                    ),
                    "fraction_with_seen_match_ge_32": float(
                        np.mean(longest_matches >= 32)
                    ),
                    "fraction_with_seen_match_ge_64": float(
                        np.mean(longest_matches >= 64)
                    ),
                    "fraction_with_seen_match_ge_128": float(
                        np.mean(longest_matches >= 128)
                    ),
                    "fraction_with_seen_match_ge_192": float(
                        np.mean(longest_matches >= 192)
                    ),
                    "fraction_exact_seen_256_window": float(
                        np.mean(longest_matches >= 256)
                    ),
                    "mean_fraction_aligned_blocks_seen": float(
                        aligned_seen_blocks.mean() / (tokens.shape[1] // BLOCK_SIZE)
                    ),
                    "mean_longest_constant_character_run": float(
                        constant_runs.mean()
                    ),
                    "max_longest_constant_character_run": int(constant_runs.max()),
                }
            )
            stamp(
                "stage=output_audit_done "
                f"label={label} nfe={nfe} unique={unique_sequences}/{len(tokens)} "
                f"longest_match_mean={longest_matches.mean():.2f} "
                f"match_ge64={np.mean(longest_matches >= 64):.3f} "
                f"aligned_seen_blocks={aligned_seen_blocks.mean() / 16:.3f}"
            )

    with (OUTPUT / "memorization_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    (OUTPUT / "memorization_audit.json").write_text(
        json.dumps(audit_rows, indent=2)
    )
    (OUTPUT / "top_seen_train_matches.json").write_text(
        json.dumps(top_matches, indent=2)
    )
    np.savez_compressed(
        OUTPUT / "memorization_per_sample.npz",
        **per_sample_arrays,
    )

    audit_by_key = {
        (row["checkpoint_label"], row["nfe_steps_per_block"]): row
        for row in audit_rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    steps = [row["nfe_steps_per_block"] for row in metrics]
    axes[0].plot(
        steps,
        [row["source_mean_nll"] for row in metrics],
        marker="o",
        label="source",
    )
    axes[0].plot(
        steps,
        [row["step90k_mean_nll"] for row in metrics],
        marker="o",
        label="90k",
    )
    axes[0].set_title("GPT-J-6B generative NLL")
    axes[0].set_ylabel("Mean NLL (lower is better)")
    axes[0].legend()

    axes[1].plot(
        steps,
        [row["source_pooled_char_entropy_nats"] for row in metrics],
        marker="o",
        label="source",
    )
    axes[1].plot(
        steps,
        [row["step90k_pooled_char_entropy_nats"] for row in metrics],
        marker="o",
        label="90k",
    )
    axes[1].set_title("Pooled character entropy")
    axes[1].set_ylabel("Nats")
    axes[1].legend()

    axes[2].plot(
        steps,
        [
            audit_by_key[("source", nfe)]["mean_longest_seen_train_substring"]
            for nfe in steps
        ],
        marker="o",
        label="source",
    )
    axes[2].plot(
        steps,
        [
            audit_by_key[("step90k", nfe)]["mean_longest_seen_train_substring"]
            for nfe in steps
        ],
        marker="o",
        label="90k",
    )
    axes[2].set_title("Exact copying from seen train prefix")
    axes[2].set_ylabel("Mean longest matching substring (chars)")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("NFE (steps per block)")
        axis.set_xticks(steps)
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Blockwise CFM B=16: source weights versus 90k fine-tune",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "source_protocol": json.loads((RUNS["source"] / "protocol.json").read_text()),
        "step90k_protocol": json.loads(
            (RUNS["step90k"] / "protocol.json").read_text()
        ),
        "suffix_automaton_states": int(automaton["states"]),
        "train_prefix_tokens": TRAIN_PREFIX_TOKENS,
        "written": [
            str(OUTPUT / name)
            for name in (
                "comparison.csv",
                "comparison.json",
                "memorization_audit.csv",
                "memorization_audit.json",
                "top_seen_train_matches.json",
                "memorization_per_sample.npz",
                "comparison.png",
            )
        ],
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    stamp(
        "stage=done "
        f"comparison={OUTPUT / 'comparison.csv'} "
        f"audit={OUTPUT / 'memorization_audit.csv'} "
        f"plot={OUTPUT / 'comparison.png'}"
    )


if __name__ == "__main__":
    main()
