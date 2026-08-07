"""BD3-LM generation without optional diagnostic reductions.

The token-producing operations and their order are copied from
``BlockDiffusionTextModel.generate``.  Only entropy, mask-history,
first-hitting-history, and their host synchronizations are omitted.
"""

from __future__ import annotations

import torch

from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerationResult,
    sample_logits_gumbel64,
    seeded_generator,
)


@torch.inference_mode()
def generate(model, config: GenerationConfig) -> GenerationResult:
    if config.length % model.block_size:
        raise ValueError("generation length must be divisible by block_size")
    steps = config.steps_per_block if config.steps_per_block is not None else config.num_steps
    if steps <= 0:
        raise ValueError("steps_per_block must be positive")
    device = next(model.parameters()).device
    generator = seeded_generator(device, config.seed)
    prefix = torch.empty((config.batch_size, 0), device=device, dtype=torch.long)
    cache = None
    nfe = 0

    for _ in range(config.length // model.block_size):
        active = torch.full(
            (config.batch_size, model.block_size),
            model.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        if model.ignore_bos and prefix.shape[1] == 0:
            if not 0 <= config.start_token_id < model.data_vocab_size:
                raise ValueError("start_token_id must be a data-vocabulary BOS token")
            active[:, 0] = config.start_token_id

        if config.first_hitting:
            time_values = torch.ones(config.batch_size, device=device)
            for _ in range(steps):
                masked = active.eq(model.mask_token_id)
                masked_counts = masked.sum(-1)
                if not masked.any():
                    break
                uniform = torch.rand(
                    (config.batch_size,), device=device, generator=generator
                ).clamp_min(torch.finfo(torch.float32).tiny)
                time_values = time_values * uniform.pow(
                    masked_counts.clamp_min(1).to(torch.float32).reciprocal()
                )
                logits = model._active_block_logits(prefix, active, time_values, cache)
                proposals, _ = sample_logits_gumbel64(logits, config, generator)
                priorities = torch.rand(active.shape, device=device, generator=generator)
                priorities.masked_fill_(~masked, torch.inf)
                chosen = priorities.argmin(-1, keepdim=True)
                reveal = torch.zeros_like(masked).scatter_(1, chosen, True) & masked
                active = torch.where(reveal, proposals, active)
                nfe += 1
        else:
            schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
            for t_tensor, s_tensor in zip(schedule[:-1], schedule[1:]):
                t, s = float(t_tensor.item()), float(s_tensor.item())
                logits = model._active_block_logits(prefix, active, t, cache)
                proposals, _ = sample_logits_gumbel64(logits, config, generator)
                masked = active.eq(model.mask_token_id)
                reveal_probability = 1.0 if s == 0 else (t - s) / t
                reveal = (
                    torch.rand(active.shape, device=device, generator=generator)
                    < reveal_probability
                )
                active = torch.where(masked & reveal, proposals, active)
                nfe += 1

        if active.eq(model.mask_token_id).any():
            if config.first_hitting:
                raise RuntimeError("first-hitting sampling requires steps_per_block >= block_size")
            raise RuntimeError("exact final block transition left masked tokens")
        prefix_length = prefix.shape[1]
        prefix = torch.cat((prefix, active), dim=1)
        if config.use_kv_cache:
            cache = model._append_cache(active, prefix_length, cache)

    return GenerationResult(
        tokens=prefix,
        diagnostics={
            "num_function_evaluations": nfe,
            "steps_per_block": steps,
            "sampler": "first_hitting" if config.first_hitting else "ancestral",
            "kv_cache": config.use_kv_cache,
            "seed": config.seed,
            "diagnostics_collected": False,
            "one_step_is_exact_final_transition": not config.first_hitting and steps == 1,
        },
    )
