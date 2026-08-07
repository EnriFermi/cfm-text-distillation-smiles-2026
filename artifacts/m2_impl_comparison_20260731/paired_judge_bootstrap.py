"""Paired GPT-J perplexity comparison for the two NFE=4 sample dumps.

The NLL/masking calculation intentionally mirrors ``eval.score_texts``.  In
addition to the corpus perplexity, this script preserves one NLL/token-count
pair per generated sequence and bootstraps the paired corpus-level difference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers


def load_texts(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if len(payload["points"]) != 1:
        raise ValueError(f"expected one point in {path}")
    return payload["points"][0]["texts"]


def per_sequence_nll(
    texts: list[str], tokenizer, model, batch_size: int, context_size: int, label: str
) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=context_size,
        add_special_tokens=False,
    )
    ids = encoded["input_ids"]
    attn = encoded["attention_mask"]
    nll_sums: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, ids.size(0), batch_size):
            chunk = ids[start : start + batch_size].to(model.device)
            mask = attn[start : start + batch_size].to(model.device)
            logits = model(chunk, attention_mask=mask).logits.transpose(-1, -2)
            nll = F.cross_entropy(
                logits[..., :-1].float(), chunk[..., 1:], reduction="none"
            )
            first_eos = (chunk == tokenizer.eos_token_id).cumsum(-1) == 1
            valid = (
                first_eos[..., 1:] + (chunk != tokenizer.eos_token_id)[..., 1:]
            ).float()
            nll_sums.append((nll * valid).sum(-1).cpu())
            token_counts.append(valid.sum(-1).cpu())
            done = min(start + batch_size, ids.size(0))
            print(
                f"[{label}] {done}/{ids.size(0)} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    return torch.cat(nll_sums).numpy(), torch.cat(token_counts).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    args = parser.parse_args()

    current_texts = load_texts(args.current)
    upstream_texts = load_texts(args.upstream)
    if len(current_texts) != len(upstream_texts):
        raise ValueError("paired dumps have different sample counts")

    print(
        "[config] "
        + json.dumps(
            {
                "current": str(args.current),
                "upstream": str(args.upstream),
                "judge": args.judge,
                "device": args.device,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "context_size": args.context_size,
                "n_pairs": len(current_texts),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "out": str(args.out),
            },
            indent=2,
        ),
        flush=True,
    )
    print("[stage] loading judge", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.judge)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.judge, torch_dtype=getattr(torch, args.dtype)
    ).eval().to(args.device)

    print("[stage] current per-sequence NLL", flush=True)
    current_nll, current_tokens = per_sequence_nll(
        current_texts, tokenizer, model, args.batch_size, args.context_size, "current"
    )
    print("[stage] upstream per-sequence NLL", flush=True)
    upstream_nll, upstream_tokens = per_sequence_nll(
        upstream_texts, tokenizer, model, args.batch_size, args.context_size, "upstream"
    )

    def corpus_log_ppl(nll: np.ndarray, counts: np.ndarray) -> float:
        return float(nll.sum(dtype=np.float64) / counts.sum(dtype=np.float64))

    current_log_ppl = corpus_log_ppl(current_nll, current_tokens)
    upstream_log_ppl = corpus_log_ppl(upstream_nll, upstream_tokens)
    observed_delta = upstream_log_ppl - current_log_ppl

    print("[stage] paired bootstrap", flush=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    deltas = np.empty(args.bootstrap_replicates, dtype=np.float64)
    n_pairs = len(current_nll)
    chunk_size = 1_000
    for start in range(0, args.bootstrap_replicates, chunk_size):
        size = min(chunk_size, args.bootstrap_replicates - start)
        indices = rng.integers(0, n_pairs, size=(size, n_pairs))
        cur_lp = current_nll[indices].sum(axis=1) / current_tokens[indices].sum(axis=1)
        up_lp = upstream_nll[indices].sum(axis=1) / upstream_tokens[indices].sum(axis=1)
        deltas[start : start + size] = up_lp - cur_lp

    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    payload = {
        "protocol": {
            "judge": args.judge,
            "judge_commit": getattr(model.config, "_commit_hash", None),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": str(model.device),
            "dtype": str(next(model.parameters()).dtype),
            "batch_size": args.batch_size,
            "context_size": args.context_size,
            "n_pairs": n_pairs,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "current": {
            "total_nll": float(current_nll.sum(dtype=np.float64)),
            "total_tokens": int(current_tokens.sum()),
            "log_ppl": current_log_ppl,
            "ppl": math.exp(current_log_ppl),
        },
        "upstream": {
            "total_nll": float(upstream_nll.sum(dtype=np.float64)),
            "total_tokens": int(upstream_tokens.sum()),
            "log_ppl": upstream_log_ppl,
            "ppl": math.exp(upstream_log_ppl),
        },
        "paired_difference": {
            "definition": "upstream minus current",
            "log_ppl": observed_delta,
            "ppl": math.exp(upstream_log_ppl) - math.exp(current_log_ppl),
            "relative_ppl": math.exp(observed_delta) - 1.0,
            "bootstrap_95pct_ci_log_ppl": [float(ci_low), float(ci_high)],
            "bootstrap_95pct_ci_relative_ppl": [
                math.exp(float(ci_low)) - 1.0,
                math.exp(float(ci_high)) - 1.0,
            ],
            "bootstrap_probability_upstream_lower": float(np.mean(deltas < 0)),
        },
        "per_sequence": {
            "current_nll": current_nll.tolist(),
            "current_tokens": current_tokens.astype(int).tolist(),
            "upstream_nll": upstream_nll.tolist(),
            "upstream_tokens": upstream_tokens.astype(int).tolist(),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[done] current_ppl={payload['current']['ppl']:.6f} "
        f"upstream_ppl={payload['upstream']['ppl']:.6f} "
        f"relative_delta={payload['paired_difference']['relative_ppl']:.3%} "
        f"ci=[{payload['paired_difference']['bootstrap_95pct_ci_relative_ppl'][0]:.3%},"
        f"{payload['paired_difference']['bootstrap_95pct_ci_relative_ppl'][1]:.3%}] "
        f"out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
