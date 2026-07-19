"""Blockwise (semi-autoregressive) sampling for BCFM.

- **Inference-time (M2)**: ``blockwise_sample`` — generates block-by-block with a
  **clean-prefix KV cache** on the pretrained full-sequence head (``duo.DIT``).
  Each jump scores only the current block of size ``B``; after discretization the
  block is encoded at ``s=t=1`` into the cache when a later block needs it
  (denoising-step K/V are not kept).
  Total length is ``num_blocks * block_size`` (any multiple of ``B``, not hard-wired
  to the training length).
- **Training-time (M3)**: ``block_causal_sample`` — block-causal ``BlockDIT`` with
  the same clean-prefix cache API (``use_kv_cache=False`` keeps the full ``2L`` path).
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
    num_blocks: int | None = None,
    schedule: list[tuple[float, float]] | None = None,
    discretize: Literal["argmax", "sample"] = "argmax",
    gold_prefix: Tensor | None = None,
) -> Tensor:
    """Inference-time BCFM (M2) over a pretrained full-sequence CFM.

    Each jump runs ``net.forward_block`` on the current block only, attending to a KV
    cache of the finalized clean prefix. After a non-final block is discretized, it is
    encoded once at ``s=t=1`` and appended to that cache. Generation length is
    ``num_blocks * block_size`` (or ``length`` if given); it need not equal the training
    length.

    M2 intentionally has no masked full-sequence fallback: the underlying M1 ``DIT``
    has scalar time conditioning, so the legacy doubled-stream implementation could not
    represent a clean prefix at ``s=t=1`` and a noisy block at ``(s, t)`` in one call.
    """
    K = module.in_shape[-1]
    device = module.device
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]

    if num_blocks is not None:
        n_blocks = int(num_blocks)
        L = n_blocks * block_size
    else:
        L = int(length if length is not None else module.in_shape[0])
        if L % block_size != 0:
            raise ValueError(f"length {L} not divisible by block_size {block_size}")
        n_blocks = L // block_size

    return _blockwise_sample_cached(
        module,
        block_size=block_size,
        n_blocks=n_blocks,
        batch_size=batch_size,
        K=K,
        device=device,
        sched=sched,
        discretize=discretize,
        gold_prefix=gold_prefix,
    )


def _blockwise_sample_cached(
    module,
    *,
    block_size: int,
    n_blocks: int,
    batch_size: int,
    K: int,
    device,
    sched: list[tuple[float, float]],
    discretize: str,
    gold_prefix: Tensor | None,
) -> Tensor:
    from semicat.net.duo import cache_seq_len, empty_kv_cache

    net = module.net
    if not hasattr(net, "forward_block") or not hasattr(net, "encode_clean"):
        raise TypeError(
            "M2 blockwise inference requires net.encode_clean / net.forward_block "
            f"(got {type(net).__name__})"
        )

    L = n_blocks * block_size
    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    kv_cache = empty_kv_cache(len(net.blocks))

    for b in range(n_blocks):
        lo, hi = b * block_size, (b + 1) * block_size
        positions = torch.arange(lo, hi, device=device)
        z_blk = module.prior((batch_size, block_size, K), device=device)
        for s, t in sched:
            s_b = torch.full((batch_size,), s, device=device)
            t_b = torch.full((batch_size,), t, device=device)
            q_blk = net.forward_block(z_blk, s_b, t_b, positions, kv_cache).softmax(dim=-1)
            z_blk = _euler_step(z_blk, q_blk, s, t)
        block = _discretize(z_blk, discretize)
        tokens[:, lo:hi] = block
        # The final block has no successor, so building its K/V cache is pure overhead.
        if b + 1 < n_blocks:
            cond_blk = gold_prefix[:, lo:hi].to(device) if gold_prefix is not None else block
            clean_oh = F.one_hot(cond_blk, K).to(z_blk.dtype)
            kv_cache = net.encode_clean(clean_oh, positions, kv_cache)
            assert cache_seq_len(kv_cache) == hi

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
    use_kv_cache: bool = True,
) -> Tensor:
    """Block-causal sampler with optional clean-prefix KV cache (M3 / ``BlockDIT``).

    Generates block by block through a block-causal ``BlockDIT``. For block ``b`` we run
    the flow-map jumps on a fresh prior block while conditioning on the finalized clean
    prefix. With ``use_kv_cache=True`` (default) only the current noisy block is scored
    against cached clean-prefix K/V; ``False`` falls back to a full ``2L`` masked forward
    (useful for equivalence tests).
    """
    from block.block_dit import cache_seq_len, empty_kv_cache
    from block.mask import block_causal_mask

    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size != 0:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]
    net = module.net

    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)

    if use_kv_cache:
        kv_cache = empty_kv_cache(len(net.blocks))
        for b in range(L // block_size):
            lo, hi = b * block_size, (b + 1) * block_size
            positions = torch.arange(lo, hi, device=device)
            z_blk = module.prior((batch_size, block_size, K), device=device)
            for s, t in sched:
                s_tok = torch.full((batch_size, block_size), s, device=device)
                t_tok = torch.full((batch_size, block_size), t, device=device)
                q_blk = net.forward_block(z_blk, s_tok, t_tok, positions, kv_cache).softmax(dim=-1)
                z_blk = _euler_step(z_blk, q_blk, s, t)
            block = _discretize(z_blk, discretize)
            tokens[:, lo:hi] = block
            # As in M2, the terminal block has no consumer for its cache.
            if b + 1 < L // block_size:
                clean_oh = F.one_hot(block, K).to(z_blk.dtype)
                kv_cache = net.encode_clean(clean_oh, positions, kv_cache)
                assert cache_seq_len(kv_cache) == hi
        return tokens

    mask = block_causal_mask(L, block_size, device)
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
            q_blk = net(x, s_full, t_full, attn_mask=mask)[:, L + lo:L + hi].softmax(dim=-1)
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
