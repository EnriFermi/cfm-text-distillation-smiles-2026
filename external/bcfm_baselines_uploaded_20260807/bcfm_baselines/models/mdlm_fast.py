"""Maximum-speed inference-only sampler for the uploaded TinyStories MDLM."""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bcfm_baselines.models.bd3lm_fast import (
    _apply_cached_rotary,
    _finish_block,
    _rotary_tables,
)
from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerationResult,
    sample_logits,
    seeded_generator,
)


class CompiledMDLMDenoiser(nn.Module):
    """Full-sequence MDLM denoiser with precomputed condition and rotary tables."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: Tensor,
        condition: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) -> Tensor:
        backbone = self.model.backbone
        x = backbone.token_embedding(tokens)
        batch, sequence_length, _ = x.shape
        heads = backbone.config.num_attention_heads
        head_dim = backbone.config.hidden_size // heads
        for block in backbone.blocks:
            modulation = block.modulation(condition)
            if modulation.ndim == 2:
                modulation = modulation[:, None, :]
            modulation = modulation.chunk(6, dim=-1)
            shift_a, scale_a, *_ = modulation
            residual = x
            hidden = block._modulate(block.norm1(x), shift_a, scale_a)
            qkv = block.qkv(hidden).view(
                batch, sequence_length, 3, heads, head_dim,
            )
            query, key, value = qkv.unbind(dim=2)
            query = _apply_cached_rotary(
                query.transpose(1, 2), rotary_cos, rotary_sin,
            )
            key = _apply_cached_rotary(
                key.transpose(1, 2), rotary_cos, rotary_sin,
            )
            attention = F.scaled_dot_product_attention(
                query,
                key,
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            x = _finish_block(block, x, residual, attention, modulation)
        logits = backbone.output(backbone.final_norm(x))
        return logits[..., : self.model.data_vocab_size]


_COMPILED_DENOISERS: dict[tuple[int, str], nn.Module] = {}


def _compile_mode() -> str:
    mode = os.environ.get("BCFM_INFERENCE_COMPILE_MODE", "reduce-overhead")
    if mode not in {"reduce-overhead", "max-autotune"}:
        raise ValueError(f"unsupported BCFM_INFERENCE_COMPILE_MODE={mode!r}")
    return mode


def compiled_mdlm_denoiser(model: nn.Module) -> nn.Module:
    mode = _compile_mode()
    key = (id(model), mode)
    scorer = _COMPILED_DENOISERS.get(key)
    if scorer is None:
        scorer = torch.compile(
            CompiledMDLMDenoiser(model),
            fullgraph=True,
            mode=mode,
        )
        _COMPILED_DENOISERS[key] = scorer
    return scorer


@torch.inference_mode()
def generate_fast_mdlm(
    model,
    generation_config: GenerationConfig,
    *,
    compile_steps: bool = True,
) -> GenerationResult:
    """Canonical MDLM sampling without metric-only per-step diagnostics."""
    config = generation_config
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if model.time_conditioning:
        raise ValueError("fast MDLM sampler currently targets time_conditioning=False")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generator = seeded_generator(device, config.seed)
    tokens = torch.full(
        (config.batch_size, config.length),
        model.mask_token_id,
        device=device,
        dtype=torch.long,
    )
    zero_time = torch.zeros(config.batch_size, device=device)
    condition = F.silu(model.backbone.time_embedding(zero_time))
    rotary_cos, rotary_sin = _rotary_tables(model, device=device, dtype=dtype)
    scorer: nn.Module = (
        compiled_mdlm_denoiser(model)
        if compile_steps else CompiledMDLMDenoiser(model)
    )
    schedule = torch.linspace(
        1.0, config.sampling_epsilon, config.num_steps + 1,
        dtype=torch.float32,
    ).tolist()
    cached_logits: Tensor | None = None
    nfe = 0
    cache_hits = 0
    for t_value, s_value in zip(schedule[:-1], schedule[1:]):
        if cached_logits is None:
            logits = scorer(tokens, condition, rotary_cos, rotary_sin)
            nfe += 1
        else:
            logits = cached_logits
            cache_hits += 1
        proposals, _ = sample_logits(logits, config, generator)
        masked = tokens.eq(model.mask_token_id)
        reveal_probability = (t_value - s_value) / t_value
        reveal = torch.rand(
            tokens.shape, device=device, generator=generator,
        ) < reveal_probability
        next_tokens = torch.where(masked & reveal, proposals, tokens)
        changed = not torch.equal(next_tokens, tokens)
        tokens = next_tokens
        if config.cache_denoiser_outputs and not changed:
            cached_logits = logits
        else:
            cached_logits = None
    if config.noise_removal:
        masked = tokens.eq(model.mask_token_id)
        logits = scorer(tokens, condition, rotary_cos, rotary_sin)
        tokens = torch.where(masked, logits.argmax(-1), tokens)
        nfe += 1
    return GenerationResult(
        tokens=tokens,
        diagnostics={
            "num_function_evaluations": nfe,
            "diffusion_steps": config.num_steps,
            "sampler": "mdlm",
            "compiled": compile_steps,
            "denoiser_output_cache": config.cache_denoiser_outputs,
            "denoiser_cache_hits": cache_hits,
            "seed": config.seed,
        },
    )
