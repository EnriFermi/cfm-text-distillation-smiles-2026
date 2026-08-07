"""`apply_rotary_emb_torch` copied from flash-attn's own pure-torch reference path."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    """x: (batch, seqlen, nheads, headdim); cos/sin: (seqlen, rotary_dim / 2)."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.shape[1]
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    # einops `repeat(cos, "s d -> s 1 (2 d)")`
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_emb_qkv_(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kw):
    """Not used by this repo's forward path; fail loudly rather than silently differ."""
    raise NotImplementedError(
        "flash_attn shim implements only apply_rotary_emb_torch"
    )
