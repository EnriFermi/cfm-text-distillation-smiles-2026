"""Maximum-speed, inference-only sampler for the uploaded TinyStories BD3-LM.

The checkpoint and probabilistic sampler are unchanged.  This execution path:

* evaluates only the active block against cached finalized-prefix K/V;
* commits the previous clean block during the next block's first denoising pass;
* skips a cache commit for the final block;
* compiles the active and fused transition graphs with CUDA Graphs;
* keeps mask updates and sampling on GPU and records no per-step diagnostics.

The module is intentionally separate from ``bd3lm.py`` so the canonical uploaded
implementation remains available as an auditable baseline.
"""

from __future__ import annotations

import math
import os
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerationResult,
    sample_logits,
    seeded_generator,
)


def _apply_cached_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def _finish_block(block, x: Tensor, residual: Tensor, attention: Tensor, modulation) -> Tensor:
    _, _, gate_a, shift_m, scale_m, gate_m = modulation
    hidden = attention.transpose(1, 2).flatten(2)
    x = residual + block.dropout(block.attention_output(hidden) * gate_a)
    return x + block.dropout(
        block.mlp(block._modulate(block.norm2(x), shift_m, scale_m)) * gate_m
    )


class DynamicBD3ActiveStep(nn.Module):
    """Score one masked block against an exactly sized clean-prefix cache."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        active: Tensor,
        condition: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        clean_keys: tuple[Tensor, ...],
        clean_values: tuple[Tensor, ...],
    ) -> Tensor:
        backbone = self.model.backbone
        x = backbone.token_embedding(active)
        batch, tokens, _ = x.shape
        heads = backbone.config.num_attention_heads
        head_dim = backbone.config.hidden_size // heads
        for layer_index, block in enumerate(backbone.blocks):
            modulation = block.modulation(condition)
            if modulation.ndim == 2:
                modulation = modulation[:, None, :]
            modulation = modulation.chunk(6, dim=-1)
            shift_a, scale_a, *_ = modulation
            residual = x
            hidden = block._modulate(block.norm1(x), shift_a, scale_a)
            qkv = block.qkv(hidden).view(batch, tokens, 3, heads, head_dim)
            query, key, value = qkv.unbind(dim=2)
            query = _apply_cached_rotary(
                query.transpose(1, 2), rotary_cos, rotary_sin,
            )
            key = _apply_cached_rotary(
                key.transpose(1, 2), rotary_cos, rotary_sin,
            )
            value = value.transpose(1, 2)
            key = torch.cat((clean_keys[layer_index], key), dim=-2)
            value = torch.cat((clean_values[layer_index], value), dim=-2)
            attention = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False,
            )
            x = _finish_block(block, x, residual, attention, modulation)
        logits = backbone.output(backbone.final_norm(x))
        return logits[..., : self.model.data_vocab_size]


class DynamicBD3FusedTransition(nn.Module):
    """Commit clean previous K/V while scoring the next masked block."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        clean_previous: Tensor,
        active: Tensor,
        condition: Tensor,
        previous_cos: Tensor,
        previous_sin: Tensor,
        active_cos: Tensor,
        active_sin: Tensor,
        clean_keys: tuple[Tensor, ...],
        clean_values: tuple[Tensor, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        backbone = self.model.backbone
        block_size = active.shape[1]
        combined = torch.cat((clean_previous, active), dim=1)
        x = backbone.token_embedding(combined)
        batch, tokens, _ = x.shape
        heads = backbone.config.num_attention_heads
        head_dim = backbone.config.hidden_size // heads
        rotary_cos = torch.cat((previous_cos, active_cos), dim=-2)
        rotary_sin = torch.cat((previous_sin, active_sin), dim=-2)
        new_keys: list[Tensor] = []
        new_values: list[Tensor] = []

        for layer_index, block in enumerate(backbone.blocks):
            modulation = block.modulation(condition)
            if modulation.ndim == 2:
                modulation = modulation[:, None, :]
            modulation = modulation.chunk(6, dim=-1)
            shift_a, scale_a, *_ = modulation
            residual = x
            hidden = block._modulate(block.norm1(x), shift_a, scale_a)
            qkv = block.qkv(hidden).view(batch, tokens, 3, heads, head_dim)
            query, key, value = qkv.unbind(dim=2)
            query = _apply_cached_rotary(
                query.transpose(1, 2), rotary_cos, rotary_sin,
            )
            key = _apply_cached_rotary(
                key.transpose(1, 2), rotary_cos, rotary_sin,
            )
            value = value.transpose(1, 2)
            previous_key = key[..., :block_size, :]
            previous_value = value[..., :block_size, :]
            full_key = torch.cat((clean_keys[layer_index], key), dim=-2)
            full_value = torch.cat((clean_values[layer_index], value), dim=-2)
            clean_visible = clean_keys[layer_index].shape[-2] + block_size
            clean_attention = F.scaled_dot_product_attention(
                query[..., :block_size, :],
                full_key[..., :clean_visible, :],
                full_value[..., :clean_visible, :],
                dropout_p=0.0,
                is_causal=False,
            )
            active_attention = F.scaled_dot_product_attention(
                query[..., block_size:, :],
                full_key,
                full_value,
                dropout_p=0.0,
                is_causal=False,
            )
            attention = torch.cat((clean_attention, active_attention), dim=-2)
            x = _finish_block(block, x, residual, attention, modulation)
            new_keys.append(previous_key)
            new_values.append(previous_value)

        active_hidden = backbone.final_norm(x[:, block_size:])
        logits = backbone.output(active_hidden)[..., : self.model.data_vocab_size]
        return logits, tuple(new_keys), tuple(new_values)


_ACTIVE_SCORERS: dict[tuple[int, str], nn.Module] = {}
_FUSED_SCORERS: dict[tuple[int, str], nn.Module] = {}


def _inference_compile_mode() -> str:
    mode = os.environ.get("BCFM_INFERENCE_COMPILE_MODE", "reduce-overhead")
    if mode not in {"reduce-overhead", "max-autotune"}:
        raise ValueError(f"unsupported BCFM_INFERENCE_COMPILE_MODE={mode!r}")
    return mode


def _compiled_active(model) -> nn.Module:
    mode = _inference_compile_mode()
    key = (id(model), mode)
    scorer = _ACTIVE_SCORERS.get(key)
    if scorer is None:
        scorer = torch.compile(
            DynamicBD3ActiveStep(model), fullgraph=True, mode=mode,
        )
        _ACTIVE_SCORERS[key] = scorer
    return scorer


def _compiled_transition(model) -> nn.Module:
    mode = _inference_compile_mode()
    key = (id(model), mode)
    scorer = _FUSED_SCORERS.get(key)
    if scorer is None:
        scorer = torch.compile(
            DynamicBD3FusedTransition(model), fullgraph=True, mode=mode,
        )
        _FUSED_SCORERS[key] = scorer
    return scorer


def _rotary_tables(model, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    config = model.backbone.config
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        config.rotary_base
        ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(config.max_sequence_length, device=device).float()
    angles = positions[:, None] * inv_freq[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotary_slice(table: Tensor, lo: int, hi: int) -> Tensor:
    return table[lo:hi][None, None, :, :]


@torch.inference_mode()
def generate_fast_bd3(
    model,
    generation_config: GenerationConfig,
    *,
    compile_steps: bool = True,
) -> GenerationResult:
    """Generate with the fused FP32 fast path and minimal diagnostics."""
    config = generation_config
    if model.time_conditioning:
        raise ValueError("fast BD3 sampler currently targets time_conditioning=False")
    if config.length % model.block_size:
        raise ValueError("generation length must be divisible by block_size")
    steps = int(config.steps_per_block or config.num_steps)
    if steps <= 0:
        raise ValueError("steps_per_block must be positive")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generator = seeded_generator(device, config.seed)
    batch = config.batch_size
    block_size = model.block_size
    n_blocks = config.length // block_size
    tokens = torch.empty((batch, config.length), device=device, dtype=torch.long)
    empty_shape = (
        batch,
        model.backbone.config.num_attention_heads,
        0,
        model.backbone.config.hidden_size // model.backbone.config.num_attention_heads,
    )
    clean_keys: tuple[Tensor, ...] = tuple(
        torch.empty(empty_shape, device=device, dtype=dtype)
        for _ in model.backbone.blocks
    )
    clean_values: tuple[Tensor, ...] = tuple(torch.empty_like(item) for item in clean_keys)
    zero_time = torch.zeros(batch, device=device)
    condition = F.silu(model.backbone.time_embedding(zero_time))
    rotary_cos, rotary_sin = _rotary_tables(model, device=device, dtype=dtype)
    active_scorer: nn.Module = (
        _compiled_active(model) if compile_steps else DynamicBD3ActiveStep(model)
    )
    transition_scorer: nn.Module = (
        _compiled_transition(model)
        if compile_steps else DynamicBD3FusedTransition(model)
    )
    previous_clean: Tensor | None = None

    if config.first_hitting:
        if steps < block_size:
            raise ValueError("first-hitting requires steps_per_block >= block_size")
        schedule: list[tuple[float, float]] | None = None
    else:
        grid = torch.linspace(1.0, 0.0, steps + 1).tolist()
        schedule = list(zip(grid[:-1], grid[1:]))

    for block_index in range(n_blocks):
        lo = block_index * block_size
        hi = lo + block_size
        cos = _rotary_slice(rotary_cos, lo, hi)
        sin = _rotary_slice(rotary_sin, lo, hi)
        active = torch.full(
            (batch, block_size), model.mask_token_id, device=device, dtype=torch.long,
        )
        first_hitting_time = torch.ones(batch, device=device)
        iterations = block_size if config.first_hitting else steps
        for step_index in range(iterations):
            if config.first_hitting:
                # Preserve the canonical sampler's RNG stream even though this
                # checkpoint has time_conditioning=False.  We intentionally omit
                # only the diagnostic CPU copy/synchronization.
                masked_for_time = active.eq(model.mask_token_id)
                masked_counts = masked_for_time.sum(-1)
                uniform = torch.rand(
                    (batch,), device=device, generator=generator,
                ).clamp_min(torch.finfo(torch.float32).tiny)
                first_hitting_time = first_hitting_time * uniform.pow(
                    masked_counts.clamp_min(1).to(torch.float32).reciprocal()
                )
            if block_index and step_index == 0:
                if previous_clean is None:
                    raise RuntimeError("missing previous finalized block")
                previous_cos = _rotary_slice(rotary_cos, lo - block_size, lo)
                previous_sin = _rotary_slice(rotary_sin, lo - block_size, lo)
                logits, new_keys, new_values = transition_scorer(
                    previous_clean,
                    active,
                    condition,
                    previous_cos,
                    previous_sin,
                    cos,
                    sin,
                    clean_keys,
                    clean_values,
                )
                retained_keys = tuple(item.clone() for item in new_keys)
                retained_values = tuple(item.clone() for item in new_values)
                clean_keys = tuple(
                    torch.cat((old, new), dim=-2)
                    for old, new in zip(clean_keys, retained_keys)
                )
                clean_values = tuple(
                    torch.cat((old, new), dim=-2)
                    for old, new in zip(clean_values, retained_values)
                )
            else:
                logits = active_scorer(
                    active, condition, cos, sin, clean_keys, clean_values,
                )

            proposals, _ = sample_logits(logits, config, generator)
            masked = active.eq(model.mask_token_id)
            if config.first_hitting:
                priorities = torch.rand(
                    active.shape, device=device, generator=generator,
                ).masked_fill(~masked, torch.inf)
                chosen = priorities.argmin(-1, keepdim=True)
                reveal = torch.zeros_like(masked).scatter_(1, chosen, True)
                active = torch.where(masked & reveal, proposals, active)
            else:
                assert schedule is not None
                t_value, s_value = schedule[step_index]
                reveal_probability = 1.0 if s_value == 0 else (t_value - s_value) / t_value
                reveal = torch.rand(
                    active.shape, device=device, generator=generator,
                ) < reveal_probability
                active = torch.where(masked & reveal, proposals, active)

        tokens[:, lo:hi] = active
        if block_index + 1 < n_blocks:
            previous_clean = active

    return GenerationResult(
        tokens=tokens,
        diagnostics={
            "num_function_evaluations": n_blocks * iterations,
            "steps_per_block": steps,
            "sampler": "first_hitting" if config.first_hitting else "ancestral",
            "kv_cache": True,
            "cache_commit": "fused_with_next_first_step",
            "compiled": compile_steps,
            "seed": config.seed,
        },
    )
