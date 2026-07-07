"""Block-causal DiT for training-time BCFM (M3).

Based on ``semicat/net/duo.py::DIT`` but with three changes the block model needs:

1. **Manual masked attention** (plain-torch softmax with an additive mask) instead of
   flash/SDPA. This is what lets a block-causal mask flow through the ECLD forward-mode
   derivative: ``torch.func.jvp`` works through eager ops, whereas the upstream Triton
   JVP-SDPA kernel supports neither masks nor CPU. At text8 scale (L=256) the O(L^2)
   attention is cheap, so nothing is lost.
2. **Per-token (s, t) conditioning**: adaLN modulation varies per position, so every
   block can carry its own time pair — the clean stream sits at s=t=1, each noisy block
   at its sampled ``(s_b, t_b)``.
3. **Doubled-sequence forward**: consumes ``[x_clean ; z_noisy]`` (length 2L) with the
   block-causal mask; positions in the two streams share rotary positions (``pos % L``).

Pure torch, no flash-attn/Triton import, so it runs and is testable on CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- rotary
def _rope_tables(positions: Tensor, dim: int, base: float = 10_000.0) -> tuple[Tensor, Tensor]:
    """cos/sin tables of shape ``(len(positions), dim)`` for rotary embedding."""
    inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    ang = positions[:, None].float() * inv_freq[None, :]           # (S, dim/2)
    ang = torch.cat([ang, ang], dim=-1)                            # (S, dim)
    return ang.cos(), ang.sin()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, S, H, Dh); cos/sin: (S, Dh)
    cos, sin = cos[None, :, None, :].to(x.dtype), sin[None, :, None, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


# ----------------------------------------------------------------------- embedders
class TimestepEmbedder(nn.Module):
    """Sinusoidal-featured scalar-time embedder (mirrors duo's)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10_000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        args = t[:, None].float() * self.freqs[None]
        feats = torch.cat([args.cos(), args.sin()], dim=-1).to(self.mlp[0].weight.dtype)
        return self.mlp(feats)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # per-token: shift/scale are (B, N, D)
    return x * (1.0 + scale) + shift


# --------------------------------------------------------------------------- block
class BlockDiTLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.adaLN = nn.Linear(cond_dim, 6 * dim)
        self.adaLN.weight.data.zero_()
        self.adaLN.bias.data.zero_()

    def _attention(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        B, N, _ = x.shape
        qkv = self.attn_qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                       # each (B, N, H, Dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        scale = self.head_dim ** -0.5
        # scores (B, H, N, N) in fp32 for a stable, jvp-friendly softmax
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k).float() * scale
        if mask is not None:
            scores = scores.masked_fill(~mask[None, None], float("-inf"))
        attn = scores.softmax(dim=-1).to(v.dtype)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        return out.reshape(B, N, self.dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cond: Tensor, mask: Tensor | None) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(cond).chunk(6, dim=-1)
        x = x + gate_a * self.attn_out(self._attention(_modulate(self.norm1(x), shift_a, scale_a), cos, sin, mask))
        x = x + gate_m * self.dropout(self.mlp(_modulate(self.norm2(x), shift_m, scale_m)))
        return x


class BlockFinalLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, vocab_size)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN = nn.Linear(cond_dim, 2 * dim)
        self.adaLN.weight.data.zero_()
        self.adaLN.bias.data.zero_()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class BlockDIT(nn.Module):
    """DiT over the doubled sequence with a block-causal mask and per-token times.

    ``forward(x, s, t, attn_mask)``:
      - ``x``: ``(B, 2L, K)`` one-hot / simplex inputs, clean stream then noisy stream.
      - ``s``, ``t``: ``(B, 2L)`` per-token times (clean positions at 1.0).
      - ``attn_mask``: ``(2L, 2L)`` bool from ``block.mask.block_causal_mask``.
      - returns ``(B, 2L, K)`` logits; the caller uses the noisy half ``[:, L:]``.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        length: int,
        block_size: int,
        embed_type: str = "rms",  # accepted for config parity; embedding is a linear proj
    ):
        super().__init__()
        self.length = length
        self.vocab_size = vocab_size
        self._coeff = vocab_size ** 0.5
        self.vocab_embed = nn.Linear(vocab_size, hidden_size)
        self.s_map = TimestepEmbedder(cond_dim)
        self.t_map = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList(
            [BlockDiTLayer(hidden_size, n_heads, cond_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output_layer = BlockFinalLayer(hidden_size, vocab_size, cond_dim)

        # rotary positions repeat per stream: clean [0..L), noisy [0..L)
        positions = torch.cat([torch.arange(length), torch.arange(length)])
        cos, sin = _rope_tables(positions, hidden_size // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor, s: Tensor, t: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, N, _ = x.shape
        t_eff = t - s
        cond = (F.silu(self.s_map(s.reshape(-1)))
                + F.silu(self.t_map(t_eff.reshape(-1)))).reshape(B, N, -1)
        h = self.vocab_embed(x / self._coeff)
        for blk in self.blocks:
            h = blk(h, self.rope_cos, self.rope_sin, cond, attn_mask)
        return self.output_layer(h, cond)
