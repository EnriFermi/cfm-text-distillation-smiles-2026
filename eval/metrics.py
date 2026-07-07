"""Metrics for BCFM evaluation.

Thin wrappers over the upstream Duo-based pipeline (``semicat.metric.text_dist``) plus
the BCFM-specific metric: token entropy **per block index**, the early-warning signal
for blockwise entropy collapse.

Two granularities, both reported:
- *pooled*: one distribution over all samples' tokens at a block index. High even if
  every individual sample is degenerate (e.g. each repeats a different character), so
  it detects only *global* collapse toward the same tokens.
- *per-sample*: entropy within each sample's block, averaged over samples. This is the
  repetitive-text signal; its ceiling is ``log(block_size)``, so compare across equal
  block sizes only (and it is vacuous at block_size=1).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from semicat.metric.text_dist import TextMetrics


def gen_ppl(
    strings: list[str],
    batch_size: int = 64,
    context_size: int = 256,
    model: str = "gpt2-large",
    device: str = "cuda",
) -> float:
    """Generative perplexity under an external LM (the quality judge)."""
    return TextMetrics.compute_mean_gen_ppl(
        strings, batch_size, context_size=context_size, ppl_model=model, device=device
    )


def entropy_per_block(tokens: Tensor, block_size: int) -> list[float]:
    """Pooled token entropy (nats) at each block index. ``tokens``: ``(N, L)`` long."""
    N, L = tokens.shape
    out = []
    for b in range(L // block_size):
        pooled = tokens[:, b * block_size:(b + 1) * block_size].reshape(-1)
        _, counts = pooled.unique(return_counts=True)
        p = counts.float() / counts.sum()
        out.append(float(torch.special.entr(p).sum().item()))
    return out


def entropy_per_block_per_sample(tokens: Tensor, block_size: int) -> list[float]:
    """Mean over samples of within-sample token entropy (nats) at each block index."""
    N, L = tokens.shape
    K = int(tokens.max().item()) + 1
    out = []
    for b in range(L // block_size):
        blk = tokens[:, b * block_size:(b + 1) * block_size]
        counts = torch.zeros(N, K).scatter_add_(1, blk, torch.ones(blk.shape))
        p = counts / block_size
        out.append(float(torch.special.entr(p).sum(dim=-1).mean().item()))
    return out


def mean_entropy(tokens: Tensor) -> float:
    """Whole-sequence mean token entropy (matches upstream ``compute_mean_entropy``)."""
    return float(np.mean(entropy_per_block(tokens, tokens.shape[1])))
