"""Complete offline Text8 likelihood and sample evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch

from bcfm_baselines.checkpoint import load_checkpoint_model
from bcfm_baselines.data import create_corpus
from bcfm_baselines.models.generative_text import GenerationConfig


def sample_metrics(tokens: torch.Tensor, vocabulary_size: int) -> dict[str, float]:
    rows = tokens.cpu().tolist()
    flat = [token for row in rows for token in row]
    counts = Counter(flat)
    total = max(len(flat), 1)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    unigrams = set(flat)
    bigrams = {(row[index], row[index + 1]) for row in rows for index in range(len(row) - 1)}
    total_bigrams = sum(max(len(row) - 1, 0) for row in rows)
    adjacent_repeats = sum(
        row[index] == row[index - 1] for row in rows for index in range(1, len(row))
    )
    return {
        "sample_unigram_entropy_nats": entropy,
        "sample_distinct_1": len(unigrams) / total,
        "sample_distinct_2": len(bigrams) / max(total_bigrams, 1),
        "sample_adjacent_repetition_rate": adjacent_repeats / max(total_bigrams, 1),
        "sample_vocabulary_coverage": len(unigrams) / vocabulary_size,
    }


@torch.no_grad()
def evaluate(args) -> dict:
    model, payload = load_checkpoint_model(args.checkpoint, args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = payload["config"]
    corpus = create_corpus(config)
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for batch in corpus.sequential_batches("test", args.eval_batch_size, args.eval_batches):
        if model.model_type == "bd3lm":
            metrics = model.compute_loss(
                batch.to(args.device), mask_rate_bounds=(float(model.min_time), 1.0)
            )
        else:
            metrics = model.compute_loss(batch.to(args.device))
        token_count = float(metrics["token_count"].item())
        for key, value in metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                if key == "token_count":
                    totals[key] = totals.get(key, 0.0) + float(value.item())
                else:
                    totals[key] = totals.get(key, 0.0) + float(value.item()) * token_count
                    weights[key] = weights.get(key, 0.0) + token_count
    if not totals:
        raise RuntimeError("test split produced no batches")
    likelihood = {
        f"test_{key}": value if key == "token_count" else value / weights[key]
        for key, value in totals.items()
    }
    if model.model_type == "ar":
        likelihood["test_perplexity"] = math.exp(likelihood["test_nll"])
        likelihood["likelihood_metric_type"] = "exact_autoregressive_nll"
    else:
        likelihood["test_epsilon_truncated_nelbo_perplexity_estimate"] = math.exp(
            likelihood["test_nelbo"]
        )
        likelihood["likelihood_metric_type"] = (
            "single_sample_epsilon_truncated_continuous_time_nelbo"
        )
        likelihood["likelihood_time_min"] = float(model.min_time)
        likelihood["likelihood_is_strict_upper_bound"] = False

    sampling = dict(config["sampling"])
    sampling["batch_size"] = args.num_samples
    sampling["seed"] = args.seed
    if args.num_steps is not None:
        sampling["num_steps"] = args.num_steps
    if args.steps_per_block is not None:
        sampling["steps_per_block"] = args.steps_per_block
    if getattr(args, "first_hitting", None) is not None:
        sampling["first_hitting"] = args.first_hitting
    started = time.perf_counter()
    generated = model.generate(GenerationConfig(**sampling))
    sample_seconds = time.perf_counter() - started
    generated_tokens = generated.tokens.numel()
    results = {
        "model_type": model.model_type,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(payload["step"]),
        "processed_training_tokens": int(payload["processed_tokens"]),
        "parameter_counts": model.num_parameters(),
        "weights_used_for_inference": payload["weights_used_for_inference"],
        "likelihood_seed": args.seed,
        "dataset_metadata": corpus.metadata(),
        **likelihood,
        **sample_metrics(generated.tokens, corpus.vocab_size),
        "sampling_seconds": sample_seconds,
        "sampling_tokens_per_second": generated_tokens / max(sample_seconds, 1e-12),
        "sampling_diagnostics": generated.diagnostics,
        "generation_config": sampling,
        "sample_texts": corpus.decode(generated.tokens),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--steps-per-block", type=int)
    parser.add_argument(
        "--first-hitting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable the official BD3-LM first-hitting sampler",
    )
    args = parser.parse_args()
    results = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value for key, value in results.items() if key != "sample_texts"}, indent=2))


if __name__ == "__main__":
    main()
