"""Generate reproducible Text8 samples from any baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bcfm_baselines.checkpoint import load_checkpoint_model
from bcfm_baselines.data import create_corpus
from bcfm_baselines.models.generative_text import GenerationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--steps-per-block", type=int)
    parser.add_argument(
        "--first-hitting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable the official BD3-LM first-hitting sampler",
    )
    args = parser.parse_args()
    model, payload = load_checkpoint_model(args.checkpoint, args.device)
    config = payload["config"]
    sampling = dict(config["sampling"])
    updates = {
        "batch_size": args.num_samples,
        "length": args.length,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "steps_per_block": args.steps_per_block,
        "first_hitting": args.first_hitting,
    }
    sampling.update({key: value for key, value in updates.items() if value is not None})
    generation = GenerationConfig(**sampling)
    result = model.generate(generation)
    corpus = create_corpus(config)
    output = {
        "model_type": model.model_type,
        "checkpoint": str(args.checkpoint.resolve()),
        "weights_used_for_inference": payload["weights_used_for_inference"],
        "generation_config": sampling,
        "tokens": result.tokens.cpu().tolist(),
        "texts": corpus.decode(result.tokens),
        "diagnostics": result.diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("tokens", "texts")}, indent=2))


if __name__ == "__main__":
    main()
