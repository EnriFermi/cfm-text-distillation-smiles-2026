"""Block-causal attention mask for the BD3-LM doubled sequence (the M3 core).

Training packs the clean data and the noisy interpolant into one length-``2L`` sequence
``[x_clean ; z_noisy]`` and runs a single masked attention so all ``L/B`` block losses
come from one forward. Positions ``[0, L)`` are the clean stream, ``[L, 2L)`` the noisy
stream; the block of a position is ``(pos % L) // block_size``.

The mask is the one correctness-critical detail (paper §Training): a noisy block must see
only *strictly previous clean blocks* and never its own clean copy (which would leak the
answer). Rules (query → key, True = attention allowed):

- clean → clean : ``block(k) <= block(q)``   (block-causal context, KV-cacheable)
- clean → noisy : never
- noisy → clean : ``block(k) <  block(q)``    (strictly previous clean = the condition)
- noisy → noisy : ``block(k) == block(q)``    (intra-block only, bidirectional)
"""

from __future__ import annotations

import torch
from torch import Tensor


def block_causal_mask(length: int, block_size: int, device=None) -> Tensor:
    """Boolean ``(2L, 2L)`` mask, ``True`` where attention is allowed."""
    if length % block_size != 0:
        raise ValueError(f"length {length} not divisible by block_size {block_size}")
    blk = torch.arange(length, device=device) // block_size          # (L,) block id
    blk2 = torch.cat([blk, blk])                                     # (2L,)
    noisy = torch.cat([torch.zeros(length, dtype=torch.bool, device=device),
                       torch.ones(length, dtype=torch.bool, device=device)])  # (2L,)

    bq, bk = blk2[:, None], blk2[None, :]
    nq, nk = noisy[:, None], noisy[None, :]

    allowed = torch.zeros(2 * length, 2 * length, dtype=torch.bool, device=device)
    allowed |= (~nq) & (~nk) & (bk <= bq)   # clean -> clean (block-causal)
    allowed |= nq & (~nk) & (bk < bq)       # noisy -> clean (strictly previous)
    allowed |= nq & nk & (bk == bq)         # noisy -> noisy (own block)
    return allowed
