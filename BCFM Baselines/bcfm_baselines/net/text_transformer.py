"""Shared token Transformer for AR, MDLM, and BD3-LM baselines.

The architecture intentionally follows the reference CFM text backbone: pre-norm blocks,
rotary positions, GELU MLPs, 4x intermediate width, dropout, and AdaLN time
conditioning.  Attention semantics are supplied by the objective instead of
being hard-coded in separate backbones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 28
    max_sequence_length: int = 256
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    time_condition_size: int = 128
    dropout: float = 0.1
    rotary_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if min(self.vocab_size, self.max_sequence_length, self.num_layers) <= 0:
            raise ValueError("vocabulary, sequence length, and layer count must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def causal_attention_mask(query_length: int, key_length: Optional[int] = None, *, device=None) -> Tensor:
    """Return an SDPA boolean mask, accounting for a cached key prefix."""
    key_length = query_length if key_length is None else key_length
    prefix = key_length - query_length
    q = torch.arange(query_length, device=device)[:, None] + prefix
    k = torch.arange(key_length, device=device)[None, :]
    return k <= q


def block_causal_attention_mask(sequence_length: int, block_size: int, *, device=None) -> Tensor:
    """Queries attend all tokens in their block and all strictly earlier blocks."""
    if sequence_length % block_size:
        raise ValueError("sequence_length must be divisible by block_size")
    positions = torch.arange(sequence_length, device=device)
    blocks = positions // block_size
    return blocks[None, :] <= blocks[:, None]


def bd3_training_attention_mask(sequence_length: int, block_size: int, *, device=None) -> Tensor:
    """Official efficient BD3-LM mask over ``[noisy x_t, clean x_0]``.

    Noisy queries attend their own noisy block and strictly earlier clean
    blocks. Clean-copy queries attend their clean block and earlier clean
    blocks. In particular, a noisy query can never see its own clean target.
    """
    if sequence_length % block_size:
        raise ValueError("sequence_length must be divisible by block_size")
    positions = torch.arange(2 * sequence_length, device=device)
    q = positions[:, None]
    k = positions[None, :]
    q_clean = q >= sequence_length
    k_clean = k >= sequence_length
    q_block = torch.where(q_clean, (q - sequence_length) // block_size, q // block_size)
    k_block = torch.where(k_clean, (k - sequence_length) // block_size, k // block_size)
    same_half_block = (q_block == k_block) & (q_clean == k_clean)
    noisy_to_earlier_clean = (~q_clean) & k_clean & (q_block > k_block)
    clean_block_causal = q_clean & k_clean & (q_block >= k_block)
    return same_half_block | noisy_to_earlier_clean | clean_block_causal


class TimestepEmbedding(nn.Module):
    def __init__(self, output_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )
        half = frequency_size // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half) / half)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time: Tensor) -> Tensor:
        args = time.float().unsqueeze(-1) * self.frequencies
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def _apply_rotary(x: Tensor, positions: Tensor, base: float) -> Tensor:
    # x: B,H,S,D; positions: B,S or S
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("rotary head dimension must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = angles.cos().unsqueeze(1).to(x.dtype)
    sin = angles.sin().unsqueeze(1).to(x.dtype)
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


KVCache = tuple[Tensor, Tensor]


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attention_output = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, dim),
        )
        self.modulation = nn.Linear(config.time_condition_size, 6 * dim)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor],
        cache: Optional[KVCache],
        return_cache: bool,
    ) -> tuple[Tensor, Optional[KVCache]]:
        modulation = self.modulation(condition)
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, dim=-1)

        residual = x
        hidden = self._modulate(self.norm1(x), shift_a, scale_a)
        batch, query_length, _ = hidden.shape
        heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // heads
        qkv = self.qkv(hidden).view(batch, query_length, 3, heads, head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = _apply_rotary(q.transpose(1, 2), positions, self.config.rotary_base)
        k = _apply_rotary(k.transpose(1, 2), positions, self.config.rotary_base)
        v = v.transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        new_cache = (k.detach(), v.detach()) if return_cache else None
        mask = attention_mask
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
        hidden = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        hidden = hidden.transpose(1, 2).reshape(batch, query_length, -1)
        x = residual + self.dropout(self.attention_output(hidden) * gate_a)
        x = x + self.dropout(
            self.mlp(self._modulate(self.norm2(x), shift_m, scale_m)) * gate_m
        )
        return x, new_cache


class SharedTextTransformer(nn.Module):
    """One Transformer implementation shared by all three baselines."""

    def __init__(self, config: TransformerConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_embedding = TimestepEmbedding(config.time_condition_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # AdaLN-Zero keeps the initial residual backbone stable.
        for block in self.blocks:
            nn.init.zeros_(block.modulation.weight)
            nn.init.zeros_(block.modulation.bias)
        # Zero output head, as in the CFM reference's DDiTFinalLayer: training
        # starts from a uniform predictive distribution instead of the arbitrary
        # one Xavier gives, so both sides begin at the same loss.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        input_ids: Tensor,
        time: Optional[Tensor] = None,
        *,
        attention_mask: Optional[Tensor] = None,
        causal: bool = False,
        position_ids: Optional[Tensor] = None,
        cache: Optional[list[KVCache]] = None,
        return_cache: bool = False,
    ) -> tuple[Tensor, Optional[list[KVCache]]]:
        batch, query_length = input_ids.shape
        cached_length = 0 if cache is None else cache[0][0].shape[-2]
        key_length = cached_length + query_length
        if position_ids is None:
            position_ids = torch.arange(
                cached_length, key_length, device=input_ids.device
            )
        if time is None:
            time = torch.zeros(batch, device=input_ids.device)
        condition = F.silu(self.time_embedding(time))
        x = self.token_embedding(input_ids)
        if causal and attention_mask is None:
            attention_mask = causal_attention_mask(query_length, key_length, device=x.device)
        next_cache: list[KVCache] = []
        for index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[index]
            x, new_layer_cache = block(
                x, condition, position_ids, attention_mask, layer_cache, return_cache
            )
            if new_layer_cache is not None:
                next_cache.append(new_layer_cache)
        logits = self.output(self.final_norm(x))
        return logits, next_cache if return_cache else None

