"""Fast generation for the TinyStories MDLM checkpoint.

The repository sampler is intentionally diagnostic-heavy: at every diffusion
step it projects and samples all 256 positions, computes entropy over the whole
50k vocabulary, and copies several GPU reductions to the host.  Only masked
positions can affect the next state, so the fast path keeps the full-sequence
Transformer but applies the expensive vocabulary head, top-p sort, and
float64 Gumbel race only to masked positions.  It also keeps the repository's
exact denoiser-output cache when a step reveals no tokens.

Restricting random draws to active positions preserves the sampler's
distribution but not seed-for-seed token identity.  ``lean_generate`` is the
strict/token-identical path: it only removes optional diagnostics.
"""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor
from torch.nn import functional as F

from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerationResult,
    filter_logits,
    sample_logits_gumbel64,
    seeded_generator,
)


def make_hidden_body(model, *, compile_body: bool = False,
                     compile_mode: str = "default"):
    """Return the static-shape Transformer without its vocabulary projection."""
    backbone = model.backbone

    def body(input_ids: Tensor, model_time: Tensor) -> Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        condition = F.silu(backbone.time_embedding(model_time))
        hidden = backbone.token_embedding(input_ids)
        for block in backbone.blocks:
            hidden, _ = block(
                hidden,
                condition,
                positions,
                None,
                None,
                False,
            )
        return backbone.final_norm(hidden)

    if not compile_body:
        return body
    kwargs = {} if compile_mode == "default" else {"mode": compile_mode}
    return torch.compile(body, fullgraph=True, **kwargs)


def _active_samples(logits: Tensor, config: GenerationConfig,
                    generator: torch.Generator) -> Tensor:
    """The repository's float64 Gumbel/top-p draw on active rows only."""
    filtered = filter_logits(
        logits / config.temperature,
        config.top_k,
        None,
    )
    probabilities = filtered.softmax(dim=-1).to(torch.float64)
    if config.top_p is not None and config.top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(
            dim=-1, descending=True
        )
        remove = sorted_probabilities.cumsum(dim=-1) > config.top_p
        if config.top_p_include_boundary:
            remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probabilities.masked_fill_(remove, 0.0)
        probabilities.zero_().scatter_(
            -1, sorted_indices, sorted_probabilities
        )
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
    if config.strategy == "greedy":
        return probabilities.argmax(dim=-1)
    uniform = torch.rand(
        probabilities.shape,
        device=probabilities.device,
        dtype=torch.float64,
        generator=generator,
    )
    exponential = 1e-10 - (uniform + 1e-10).log()
    return (probabilities / exponential).argmax(dim=-1)


def _python_schedule(steps: int, epsilon: float) -> list[tuple[float, float]]:
    # Avoid a GPU scalar -> Python synchronization for every schedule endpoint.
    points = torch.linspace(1.0, epsilon, steps + 1, device="cpu").tolist()
    return list(zip(points[:-1], points[1:]))


@torch.inference_mode()
def fast_generate(model, config: GenerationConfig, *, body,
                  amp_dtype: torch.dtype | None = None) -> GenerationResult:
    """Distribution-equivalent MDLM sampler with active-position projection."""
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    device = next(model.parameters()).device
    generator = seeded_generator(device, config.seed)
    tokens = torch.full(
        (config.batch_size, config.length),
        model.mask_token_id,
        device=device,
        dtype=torch.long,
    )
    if model.ignore_bos:
        if not 0 <= config.start_token_id < model.data_vocab_size:
            raise ValueError("start_token_id must be a data-vocabulary BOS token")
        tokens[:, 0] = config.start_token_id

    autocast = (
        torch.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    cached_logits: Tensor | None = None
    nfe = 0
    for t, s in _python_schedule(config.num_steps, config.sampling_epsilon):
        masked = tokens.eq(model.mask_token_id)
        if not masked.any().item():
            break
        if cached_logits is None:
            time_values = torch.full(
                (config.batch_size,), t, device=device, dtype=torch.float32
            )
            with autocast:
                hidden = body(tokens, model._model_time(time_values))
                active_hidden = hidden[masked]
                logits = F.linear(
                    active_hidden,
                    model.backbone.output.weight[: model.data_vocab_size],
                    model.backbone.output.bias[: model.data_vocab_size],
                )
            nfe += 1
        else:
            logits = cached_logits

        proposals = _active_samples(logits, config, generator)
        reveal_probability = (t - s) / t
        reveal = torch.rand(
            proposals.shape, device=device, generator=generator
        ) < reveal_probability
        changed = reveal.any().item()
        active = tokens[masked]
        tokens[masked] = torch.where(reveal, proposals, active)
        if config.cache_denoiser_outputs and not changed and not model.time_conditioning:
            cached_logits = logits
        else:
            cached_logits = None

    if config.noise_removal:
        masked = tokens.eq(model.mask_token_id)
        if masked.any().item():
            time_values = torch.full(
                (config.batch_size,), config.sampling_epsilon,
                device=device, dtype=torch.float32,
            )
            with autocast:
                hidden = body(tokens, model._model_time(time_values))
                logits = F.linear(
                    hidden[masked],
                    model.backbone.output.weight[: model.data_vocab_size],
                    model.backbone.output.bias[: model.data_vocab_size],
                )
            tokens[masked] = logits.argmax(-1)
            nfe += 1

    return GenerationResult(
        tokens=tokens,
        diagnostics={
            "num_function_evaluations": nfe,
            "diffusion_steps": config.num_steps,
            "sampler": "mdlm_active_positions",
            "seed": config.seed,
            "diagnostics_collected": False,
            "active_position_rng": True,
        },
    )


@torch.inference_mode()
def lean_generate(model, config: GenerationConfig, *,
                  amp_dtype: torch.dtype | None = None) -> GenerationResult:
    """Token-identical repository sampler with diagnostics removed."""
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    device = next(model.parameters()).device
    generator = seeded_generator(device, config.seed)
    tokens = torch.full(
        (config.batch_size, config.length), model.mask_token_id,
        device=device, dtype=torch.long,
    )
    if model.ignore_bos:
        if not 0 <= config.start_token_id < model.data_vocab_size:
            raise ValueError("start_token_id must be a data-vocabulary BOS token")
        tokens[:, 0] = config.start_token_id
    autocast = (
        torch.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    cached_logits: Tensor | None = None
    nfe = 0
    for t, s in _python_schedule(config.num_steps, config.sampling_epsilon):
        if cached_logits is None:
            with autocast:
                logits = model._denoise_logits(tokens, t)
            nfe += 1
        else:
            logits = cached_logits
        proposals, _ = sample_logits_gumbel64(logits, config, generator)
        masked = tokens.eq(model.mask_token_id)
        reveal = (
            torch.rand(tokens.shape, device=device, generator=generator)
            < (t - s) / t
        )
        next_tokens = torch.where(masked & reveal, proposals, tokens)
        changed = not torch.equal(next_tokens, tokens)
        tokens = next_tokens
        cached_logits = (
            logits
            if config.cache_denoiser_outputs and not changed and not model.time_conditioning
            else None
        )
    if config.noise_removal:
        masked = tokens.eq(model.mask_token_id)
        with autocast:
            logits = model._denoise_logits(tokens, config.sampling_epsilon)
        tokens = torch.where(masked, logits.argmax(-1), tokens)
        nfe += 1
    return GenerationResult(
        tokens=tokens,
        diagnostics={
            "num_function_evaluations": nfe,
            "diffusion_steps": config.num_steps,
            "sampler": "mdlm_lean",
            "seed": config.seed,
            "diagnostics_collected": False,
            "active_position_rng": False,
        },
    )
