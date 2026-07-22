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

from functools import partial

import torch
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on old PyTorch installs
    BlockMask = object  # type: ignore[assignment,misc]
    create_block_mask = None
    FLEX_ATTENTION_AVAILABLE = False


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


def block_causal_mask_mod(
    batch_index: Tensor,
    head_index: Tensor,
    query_index: Tensor,
    key_index: Tensor,
    *,
    length: int,
    block_size: int,
) -> Tensor:
    """FlexAttention form of :func:`block_causal_mask`.

    This is the clean-first equivalent of ``models/dit.py::block_diff_mask`` in
    the official BD3-LM repository (Apache-2.0, commit
    ``1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7``). Batch/head indices are
    intentionally unused because every sample and head shares the same mask.
    """
    del batch_index, head_index
    noisy_q = query_index >= length
    noisy_k = key_index >= length
    block_q = torch.where(
        noisy_q,
        (query_index - length) // block_size,
        query_index // block_size,
    )
    block_k = torch.where(
        noisy_k,
        (key_index - length) // block_size,
        key_index // block_size,
    )
    clean_to_clean = (~noisy_q) & (~noisy_k) & (block_k <= block_q)
    noisy_to_clean = noisy_q & (~noisy_k) & (block_k < block_q)
    noisy_to_noisy = noisy_q & noisy_k & (block_k == block_q)
    return clean_to_clean | noisy_to_clean | noisy_to_noisy


def clean_block_causal_mask_mod(
    batch_index: Tensor,
    head_index: Tensor,
    query_index: Tensor,
    key_index: Tensor,
    *,
    block_size: int,
) -> Tensor:
    """Block-causal clean-stream mask used while precomputing clean KV."""
    del batch_index, head_index
    return key_index // block_size <= query_index // block_size


def create_block_causal_flex_mask(
    length: int,
    block_size: int,
    device: torch.device | str,
    *,
    kernel_block_size: int = 128,
) -> BlockMask:
    """Create the sparse doubled-sequence mask consumed by FlexAttention."""
    if not FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise RuntimeError("FlexAttention requires PyTorch >= 2.5")
    return create_block_mask(
        partial(
            block_causal_mask_mod,
            length=length,
            block_size=block_size,
        ),
        B=None,
        H=None,
        Q_LEN=2 * length,
        KV_LEN=2 * length,
        device=str(device),
        BLOCK_SIZE=kernel_block_size,
        _compile=True,
    )


def create_clean_causal_flex_mask(
    length: int,
    block_size: int,
    device: torch.device | str,
    *,
    kernel_block_size: int = 128,
) -> BlockMask:
    """Create the sparse clean-only mask used by the JVP cache path."""
    if not FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise RuntimeError("FlexAttention requires PyTorch >= 2.5")
    return create_block_mask(
        partial(clean_block_causal_mask_mod, block_size=block_size),
        B=None,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device=str(device),
        BLOCK_SIZE=kernel_block_size,
        _compile=True,
    )
