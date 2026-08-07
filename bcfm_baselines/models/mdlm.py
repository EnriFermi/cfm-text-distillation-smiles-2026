"""Absorbing-state Masked Diffusion Language Model (MDLM)."""

from __future__ import annotations

import time

import torch
from torch import Tensor
from torch.nn import functional as F

from bcfm_baselines.models.generative_text import (
    GenerationConfig,
    GenerationResult,
    GenerativeTextModel,
    batch_tokens,
    sample_time_values,
    sample_logits_gumbel64,
    seeded_generator,
)
from bcfm_baselines.net.text_transformer import SharedTextTransformer, TransformerConfig


def low_discrepancy_times(count: int, device: torch.device, generator: torch.Generator, eps: float) -> Tensor:
    return sample_time_values(
        count,
        device,
        minimum=eps,
        stratified=True,
        generator=generator,
    )


class MaskedDiffusionTextModel(GenerativeTextModel):
    model_type = "mdlm"

    def __init__(
        self,
        backbone: TransformerConfig | dict,
        data_vocab_size: int = 27,
        mask_token_id: int = 27,
        min_time: float = 1e-3,
        antithetic_time_sampling: bool = True,
        time_conditioning: bool = False,
        ignore_bos: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = SharedTextTransformer(backbone)
        self.data_vocab_size = data_vocab_size
        self.mask_token_id = mask_token_id
        self.min_time = min_time
        self.antithetic_time_sampling = antithetic_time_sampling
        self.time_conditioning = time_conditioning
        self.ignore_bos = ignore_bos

    def _clean_logits(self, logits: Tensor) -> Tensor:
        return logits[..., : self.data_vocab_size]

    def _sample_times(self, batch_size: int, device: torch.device) -> Tensor:
        return sample_time_values(
            batch_size,
            device,
            minimum=self.min_time,
            stratified=self.antithetic_time_sampling,
        )

    def _model_time(self, time: Tensor) -> Tensor:
        return time if self.time_conditioning else torch.zeros_like(time)

    def compute_loss(
        self,
        batch: Tensor | dict[str, Tensor],
        *,
        time_values: Tensor | None = None,
        corruption_uniform: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        del kwargs
        clean, valid = batch_tokens(batch)
        batch_size = clean.shape[0]
        t = self._sample_times(batch_size, clean.device) if time_values is None else time_values
        t = t.to(clean.device).clamp(min=self.min_time, max=1.0)
        noise = torch.rand(clean.shape, device=clean.device) if corruption_uniform is None else corruption_uniform
        masked = (noise < t[:, None]) & valid
        if self.ignore_bos:
            # Official MDLM/BD3 training forces the tokenizer BOS clean after
            # drawing q(x_t). At evaluation it also removes BOS from the NELBO
            # denominator; during training the zero-loss BOS remains in it.
            masked[:, 0] = False
        loss_valid = valid
        if self.ignore_bos and not self.training:
            loss_valid = valid.clone()
            loss_valid[:, 0] = False
        noisy = torch.where(masked, self.mask_token_id, clean)
        logits, _ = self.backbone(noisy, time=self._model_time(t))
        token_ce = F.cross_entropy(
            self._clean_logits(logits).transpose(1, 2), clean, reduction="none"
        )
        # Continuous-time Rao-Blackwellized NELBO for alpha(t)=1-t:
        # E[ 1{x_t=MASK} * CE(x0, p_theta) / t ].
        weighted = token_ce * masked / t[:, None]
        loss = (weighted * loss_valid).sum() / loss_valid.sum().clamp_min(1)
        return {
            "loss": loss,
            "nelbo": loss.detach(),
            "bits_per_character_estimate": loss.detach() / torch.log(loss.new_tensor(2.0)),
            "masked_fraction": masked.float().sum().detach() / loss_valid.sum().clamp_min(1),
            "token_count": loss_valid.sum().detach(),
        }

    def _denoise_logits(self, tokens: Tensor, time_value: float) -> Tensor:
        t = torch.full((tokens.shape[0],), time_value, device=tokens.device)
        logits, _ = self.backbone(tokens, time=self._model_time(t))
        return self._clean_logits(logits)

    @torch.no_grad()
    def generate(self, generation_config: GenerationConfig, **kwargs) -> GenerationResult:
        del kwargs
        config = generation_config
        if config.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        device = next(self.parameters()).device
        generator = seeded_generator(device, config.seed)
        tokens = torch.full(
            (config.batch_size, config.length),
            self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        if self.ignore_bos:
            if not 0 <= config.start_token_id < self.data_vocab_size:
                raise ValueError("start_token_id must be a data-vocabulary BOS token")
            tokens[:, 0] = config.start_token_id
        schedule = torch.linspace(
            1.0, config.sampling_epsilon, config.num_steps + 1, device=device
        )
        mask_counts = [int(tokens.eq(self.mask_token_id).sum().item())]
        entropies: list[float] = []
        started = time.perf_counter()
        cached_logits: Tensor | None = None
        nfe = 0
        for t_tensor, s_tensor in zip(schedule[:-1], schedule[1:]):
            t, s = float(t_tensor.item()), float(s_tensor.item())
            if cached_logits is None:
                logits = self._denoise_logits(tokens, t)
                nfe += 1
            else:
                logits = cached_logits
            proposals, probabilities = sample_logits_gumbel64(
                logits, config, generator
            )
            masked = tokens.eq(self.mask_token_id)
            entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
            entropies.append(float(entropy[masked].mean().item()) if masked.any() else 0.0)
            reveal_probability = (t - s) / t
            reveal = torch.rand(tokens.shape, device=device, generator=generator) < reveal_probability
            next_tokens = torch.where(masked & reveal, proposals, tokens)
            changed = not torch.equal(next_tokens, tokens)
            tokens = next_tokens
            if config.cache_denoiser_outputs and not changed and not self.time_conditioning:
                cached_logits = logits
            else:
                cached_logits = None
            mask_counts.append(int(tokens.eq(self.mask_token_id).sum().item()))
        if config.noise_removal:
            masked = tokens.eq(self.mask_token_id)
            logits = self._denoise_logits(tokens, config.sampling_epsilon)
            tokens = torch.where(masked, logits.argmax(-1), tokens)
            nfe += 1
            mask_counts.append(int(tokens.eq(self.mask_token_id).sum().item()))
        return GenerationResult(
            tokens=tokens,
            diagnostics={
                "num_function_evaluations": nfe,
                "diffusion_steps": config.num_steps,
                "masked_tokens": mask_counts,
                "token_entropy": entropies,
                "wall_clock_seconds": time.perf_counter() - started,
                "seed": config.seed,
                "schedule": "linear_survival",
                "sampling_epsilon": config.sampling_epsilon,
                "noise_removal": config.noise_removal,
                "denoiser_output_cache": config.cache_denoiser_outputs,
            },
        )
