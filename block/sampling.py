"""Blockwise (semi-autoregressive) sampling for BCFM.

- **Inference-time (M2)**: ``blockwise_sample`` — proper BD3-LM path with a doubled
  ``[clean; noisy]`` sequence and ``block_causal_mask``. Each jump runs the pretrained
  full-sequence head on ``2L`` positions; the clean stream holds finalized prefix blocks
  and is never rewritten by the flow map.
- **Training-time (M3)**: ``block_causal_sample`` — same doubled layout through a
  block-causal ``BlockDIT`` with per-token times on the noisy half.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


def uniform_schedule(steps: int) -> list[tuple[float, float]]:
    """``steps`` equal jumps covering [0, 1]: e.g. 1 -> [(0,1)], 2 -> [(0,.5),(.5,1)]."""
    grid = torch.linspace(0.0, 1.0, steps + 1).tolist()
    return list(zip(grid[:-1], grid[1:]))


def _euler_step(z_blk: Tensor, q_blk: Tensor, s: float, t: float) -> Tensor:
    return z_blk + ((t - s) / (1.0 - s + 1e-8)) * (q_blk - z_blk)


def _doubled_input(
    cond_tokens: Tensor,
    z_blk: Tensor,
    lo: int,
    hi: int,
    *,
    K: int,
    dtype: torch.dtype,
) -> Tensor:
    """Build ``[clean; noisy]`` of shape ``(batch, 2L, K)`` for one block jump."""
    batch_size, L = cond_tokens.shape
    device = cond_tokens.device
    clean = F.one_hot(cond_tokens, K).to(dtype)
    noisy = torch.zeros(batch_size, L, K, device=device, dtype=dtype)
    noisy[:, lo:hi] = z_blk
    return torch.cat([clean, noisy], dim=1)


@torch.inference_mode()
def blockwise_sample(
    module,
    block_size: int,
    steps_per_block: int,
    *,
    batch_size: int,
    length: int | None = None,
    schedule: list[tuple[float, float]] | None = None,
    discretize: Literal["argmax", "sample"] = "argmax",
    gold_prefix: Tensor | None = None,
) -> Tensor:
    """Proper inference-time BCFM (M2): BD3-LM block-causal mask over ``[clean; noisy]``.

    Wraps a pretrained *full-sequence* CFM: each in-block jump calls ``module.net`` on a
    ``2L`` sequence with ``block_causal_mask`` so the current noisy block attends only to
    finalized clean prefix blocks.
    """
    from block.mask import block_causal_mask

    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size != 0:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    mask = block_causal_mask(L, block_size, device)
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]

    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    cond = torch.zeros_like(tokens)
    for b in range(L // block_size):
        lo, hi = b * block_size, (b + 1) * block_size
        z_blk = module.prior((batch_size, block_size, K), device=device)
        for s, t in sched:
            x = _doubled_input(cond, z_blk, lo, hi, K=K, dtype=z_blk.dtype)
            s_b = torch.full((batch_size,), s, device=device)
            t_b = torch.full((batch_size,), t, device=device)
            q_blk = module.net(x, s_b, t_b, attn_mask=mask)[:, L + lo:L + hi].softmax(dim=-1)
            z_blk = _euler_step(z_blk, q_blk, s, t)
        block = _discretize(z_blk, discretize)
        tokens[:, lo:hi] = block
        cond[:, lo:hi] = gold_prefix[:, lo:hi].to(device) if gold_prefix is not None else block

    return tokens


@torch.inference_mode()
def block_causal_sample(
    module,
    block_size: int,
    steps_per_block: int,
    *,
    batch_size: int,
    length: int | None = None,
    schedule: list[tuple[float, float]] | None = None,
    discretize: Literal["argmax", "sample"] = "argmax",
) -> Tensor:
    """Sampler for the trained block-causal model (M3).

    Generates block by block through the doubled-sequence masked forward
    (``module.net`` is a ``BlockDIT``). For block ``b`` we run the flow-map jumps on a
    fresh prior block while feeding the finalized clean prefix; the block-causal mask
    guarantees block ``b``'s output depends only on clean blocks ``< b`` and its own
    noisy block.
    """
    from block.mask import block_causal_mask

    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size != 0:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    mask = block_causal_mask(L, block_size, device)
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]

    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    for b in range(L // block_size):
        lo, hi = b * block_size, (b + 1) * block_size
        z_blk = module.prior((batch_size, block_size, K), device=device)
        for s, t in sched:
            x = _doubled_input(tokens, z_blk, lo, hi, K=K, dtype=z_blk.dtype)
            s_tok = torch.zeros(batch_size, L, device=device)
            t_tok = torch.zeros(batch_size, L, device=device)
            s_tok[:, lo:hi], t_tok[:, lo:hi] = s, t
            ones = torch.ones(batch_size, L, device=device)
            s_full = torch.cat([ones, s_tok], dim=1)
            t_full = torch.cat([ones, t_tok], dim=1)
            q_blk = module.net(x, s_full, t_full, attn_mask=mask)[:, L + lo:L + hi].softmax(dim=-1)
            z_blk = _euler_step(z_blk, q_blk, s, t)
        tokens[:, lo:hi] = _discretize(z_blk, discretize)
    return tokens


def _discretize(endpoint: Tensor, how: str) -> Tensor:
    """``endpoint`` is a categorical distribution on the simplex, ``(batch, B, K)``."""
    if how == "argmax":
        return endpoint.argmax(dim=-1)
    if how == "sample":
        probs = endpoint.clamp_min(0).flatten(0, 1)
        return torch.multinomial(probs, 1).squeeze(-1).view(endpoint.shape[:-1])
    raise ValueError(f"unknown discretize mode '{how}'")
