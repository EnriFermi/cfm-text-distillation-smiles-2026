"""Block Discrete Denoising Diffusion Language Model (BD3-LM)."""

from __future__ import annotations

import math
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
from bcfm_baselines.net.text_transformer import (
    KVCache,
    SharedTextTransformer,
    TransformerConfig,
    bd3_training_attention_mask,
    block_causal_attention_mask,
)

# Mask-rate bounds reach _sample_mask either as Python floats (schedule search,
# validation) or through the float32 buffers, where 1e-3 round-trips to
# 1.0000000474974513e-3.  Endpoint tests therefore use a tolerance: an exact
# comparison would read the unclipped interval as clipped and demand at least
# one masked token in every block, which no amount of resampling delivers for a
# block whose sampled rate is epsilon.
BOUND_TOLERANCE = 1e-6


class BlockDiffusionTextModel(GenerativeTextModel):
    model_type = "bd3lm"

    def __init__(
        self,
        backbone: TransformerConfig | dict,
        block_size: int = 16,
        data_vocab_size: int = 27,
        mask_token_id: int = 27,
        min_time: float = 1e-3,
        antithetic_time_sampling: bool = True,
        time_conditioning: bool = False,
        mask_rate_min: float = 1e-3,
        mask_rate_max: float = 1.0,
        schedule_search: bool = True,
        schedule_search_widths: tuple[float, ...] | list[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
        schedule_search_delta: float = 0.05,
        schedule_search_batches: int = 100,
        resample_masked_fraction: bool = True,
        ignore_bos: bool = False,
    ) -> None:
        super().__init__()
        if block_size not in (4, 8, 16, 32):
            raise ValueError("supported BD3-LM block sizes are 4, 8, 16, and 32")
        self.backbone = SharedTextTransformer(backbone)
        self.block_size = block_size
        self.data_vocab_size = data_vocab_size
        self.mask_token_id = mask_token_id
        self.min_time = min_time
        self.antithetic_time_sampling = antithetic_time_sampling
        self.time_conditioning = time_conditioning
        if not 0 < mask_rate_min <= mask_rate_max <= 1:
            raise ValueError("mask-rate bounds must satisfy 0 < min <= max <= 1")
        if schedule_search_delta <= 0:
            raise ValueError("schedule_search_delta must be positive")
        if schedule_search_batches <= 0:
            raise ValueError("schedule_search_batches must be positive")
        self.register_buffer("mask_rate_min", torch.tensor(float(mask_rate_min)))
        self.register_buffer("mask_rate_max", torch.tensor(float(mask_rate_max)))
        self.schedule_search = schedule_search
        self.schedule_search_widths = tuple(float(width) for width in schedule_search_widths)
        self.schedule_search_delta = float(schedule_search_delta)
        self.schedule_search_batches = int(schedule_search_batches)
        self.resample_masked_fraction = resample_masked_fraction
        self.ignore_bos = ignore_bos

    def _clean_logits(self, logits: Tensor) -> Tensor:
        return logits[..., : self.data_vocab_size]

    def _sample_block_times(
        self,
        batch: int,
        blocks: int,
        device: torch.device,
        bounds: tuple[float, float] | None = None,
    ) -> Tensor:
        count = batch * blocks
        if bounds is None:
            bounds = (float(self.mask_rate_min.item()), float(self.mask_rate_max.item()))
        values = sample_time_values(
            count,
            device,
            minimum=bounds[0],
            maximum=bounds[1],
            stratified=self.antithetic_time_sampling,
        )
        return values.reshape(batch, blocks)

    def set_mask_rate_bounds(self, minimum: float, maximum: float) -> None:
        if not 0 < minimum <= maximum <= 1:
            raise ValueError("mask-rate bounds must satisfy 0 < min <= max <= 1")
        self.mask_rate_min.fill_(minimum)
        self.mask_rate_max.fill_(maximum)

    def schedule_candidates(self) -> list[tuple[float, float]]:
        """Return the paper's grid of clipped masking-rate intervals."""
        candidates = {(float(self.min_time), 1.0)}
        for width in self.schedule_search_widths:
            if not 0 < width <= 1:
                raise ValueError("schedule-search widths must be in (0, 1]")
            steps = int(round((1.0 - width) / self.schedule_search_delta))
            for index in range(steps + 1):
                lower = max(float(self.min_time), index * self.schedule_search_delta)
                upper = min(1.0, index * self.schedule_search_delta + width)
                candidates.add((round(lower, 10), round(upper, 10)))
        return sorted(candidates)

    def _unclipped_endpoints(self, bounds: tuple[float, float]) -> tuple[bool, bool]:
        """Whether each bound is the unclipped endpoint (epsilon and one)."""
        lower_is_epsilon = bounds[0] <= self.min_time or math.isclose(
            bounds[0], self.min_time, rel_tol=BOUND_TOLERANCE, abs_tol=1e-12
        )
        upper_is_one = bounds[1] >= 1.0 or math.isclose(
            bounds[1], 1.0, rel_tol=BOUND_TOLERANCE
        )
        return lower_is_epsilon, upper_is_one

    def _block_count_limits(
        self, valid_counts: Tensor, bounds: tuple[float, float]
    ) -> tuple[Tensor, Tensor]:
        """Admissible masked-token counts per block for the clipped interval.

        Counts rather than fractions keep the test exact: a block of B tokens
        can only realize multiples of 1/B, and a fraction of 0.5 must not miss
        an upper bound that a float32 buffer stores as 0.5000000149011612.
        """
        lower_is_epsilon, upper_is_one = self._unclipped_endpoints(bounds)
        minimum = (
            torch.zeros_like(valid_counts)
            if lower_is_epsilon
            else torch.ceil(bounds[0] * valid_counts - BOUND_TOLERANCE).long()
        )
        maximum = (
            valid_counts
            if upper_is_one
            else torch.floor(bounds[1] * valid_counts + BOUND_TOLERANCE).long()
        )
        # A short trailing block can leave the interval unrealizable (no integer
        # count falls inside it); the nearest attainable count is used there.
        minimum = torch.minimum(minimum, valid_counts)
        return minimum, torch.maximum(maximum, minimum)

    def _sample_mask(
        self,
        clean: Tensor,
        valid: Tensor,
        token_t: Tensor,
        bounds: tuple[float, float],
        corruption_uniform: Tensor | None,
    ) -> Tensor:
        noise = (
            torch.rand(clean.shape, device=clean.device)
            if corruption_uniform is None
            else corruption_uniform.to(clean.device)
        )
        masked = (noise < token_t) & valid
        full_interval = all(self._unclipped_endpoints(bounds))
        if (
            corruption_uniform is not None
            or not self.resample_masked_fraction
            or full_interval
        ):
            return masked

        batch, length = clean.shape
        blocks = length // self.block_size
        masked_blocks = masked.reshape(batch, blocks, self.block_size)
        valid_blocks = valid.reshape(batch, blocks, self.block_size)
        probabilities = token_t.reshape(batch, blocks, self.block_size)
        valid_counts = valid_blocks.sum(-1)
        minimum_count, maximum_count = self._block_count_limits(valid_counts, bounds)
        # Match the paper implementation exactly: redraw only invalid blocks
        # until every realized Bernoulli mask lies in the clipped interval.
        # Replacing a hard case with a fixed-count mask changes the estimator's
        # distribution and biases the variance criterion used by grid search.
        while True:
            counts = masked_blocks.sum(-1)
            invalid = (counts < minimum_count) | (counts > maximum_count)
            if not invalid.any():
                return masked_blocks.reshape_as(masked)
            redraw = torch.rand(masked_blocks.shape, device=clean.device)
            proposals = (redraw < probabilities) & valid_blocks
            masked_blocks = torch.where(invalid[..., None], proposals, masked_blocks)

    def compute_loss(
        self,
        batch: Tensor | dict[str, Tensor],
        *,
        time_values: Tensor | None = None,
        corruption_uniform: Tensor | None = None,
        mask_rate_bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        del kwargs
        clean, valid = batch_tokens(batch)
        batch_size, length = clean.shape
        if length % self.block_size:
            raise ValueError("training sequence length must be divisible by block_size")
        num_blocks = length // self.block_size
        bounds = mask_rate_bounds or (
            float(self.mask_rate_min.item()),
            float(self.mask_rate_max.item()),
        )
        block_t = (
            self._sample_block_times(batch_size, num_blocks, clean.device, bounds)
            if time_values is None
            else time_values.to(clean.device)
        )
        block_t = block_t.clamp(min=self.min_time, max=1.0)
        token_t = block_t.repeat_interleave(self.block_size, dim=1)
        masked = self._sample_mask(clean, valid, token_t, bounds, corruption_uniform)
        if self.ignore_bos:
            masked[:, 0] = False
        loss_valid = valid
        if self.ignore_bos and not self.training:
            loss_valid = valid.clone()
            loss_valid[:, 0] = False
        noisy = torch.where(masked, self.mask_token_id, clean)

        # The efficient paper formulation obtains every conditional block loss
        # in one pass over [x_t, x_0] using the specialized attention mask.
        model_input = torch.cat((noisy, clean), dim=1)
        # Clean conditioning keys are time-independent; only the noisy query
        # block receives its sampled diffusion time. This makes finalized-prefix
        # KV caching exactly equivalent to the uncached computation.
        noisy_time = token_t if self.time_conditioning else torch.zeros_like(token_t)
        model_time = torch.cat((noisy_time, torch.zeros_like(token_t)), dim=1)
        positions = torch.arange(length, device=clean.device).repeat(2)
        attention = bd3_training_attention_mask(length, self.block_size, device=clean.device)
        logits, _ = self.backbone(
            model_input,
            time=model_time,
            attention_mask=attention,
            position_ids=positions,
        )
        logits = self._clean_logits(logits[:, :length])
        token_ce = F.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
        weighted = token_ce * masked / token_t
        loss = (weighted * loss_valid).sum() / loss_valid.sum().clamp_min(1)
        block_valid = valid.reshape(batch_size, num_blocks, self.block_size).sum(-1)
        per_block_nelbo = (
            (weighted * valid)
            .reshape(batch_size, num_blocks, self.block_size)
            .sum(-1)
            / block_valid.clamp_min(1)
        )
        return {
            "loss": loss,
            "nelbo": loss.detach(),
            "bits_per_character_estimate": loss.detach() / torch.log(loss.new_tensor(2.0)),
            "masked_fraction": masked.float().sum().detach() / loss_valid.sum().clamp_min(1),
            "token_count": loss_valid.sum().detach(),
            "per_block_nelbo": per_block_nelbo.detach(),
        }

    def _active_block_logits(
        self,
        prefix: Tensor,
        active: Tensor,
        time_value: float | Tensor,
        cache: list[KVCache] | None,
    ) -> Tensor:
        batch = active.shape[0]
        if isinstance(time_value, Tensor):
            active_time = time_value.to(device=active.device, dtype=torch.float32).reshape(batch)
        else:
            active_time = torch.full((batch,), time_value, device=active.device)
        if not self.time_conditioning:
            active_time = torch.zeros_like(active_time)
        if cache is not None:
            positions = torch.arange(
                prefix.shape[1], prefix.shape[1] + active.shape[1], device=active.device
            )
            logits, _ = self.backbone(
                active,
                time=active_time,
                position_ids=positions,
                cache=cache,
            )
            return self._clean_logits(logits)
        combined = torch.cat((prefix, active), dim=1)
        attention = block_causal_attention_mask(
            combined.shape[1], self.block_size, device=active.device
        )
        combined_time = torch.cat(
            (
                torch.zeros((batch, prefix.shape[1]), device=active.device),
                active_time[:, None].expand(-1, active.shape[1]),
            ),
            dim=1,
        )
        logits, _ = self.backbone(
            combined,
            time=combined_time,
            attention_mask=attention,
        )
        return self._clean_logits(logits[:, -active.shape[1] :])

    def _append_cache(
        self,
        finalized: Tensor,
        prefix_length: int,
        cache: list[KVCache] | None,
    ) -> list[KVCache]:
        positions = torch.arange(
            prefix_length, prefix_length + finalized.shape[1], device=finalized.device
        )
        _, next_cache = self.backbone(
            finalized,
            time=torch.zeros(finalized.shape[0], device=finalized.device),
            position_ids=positions,
            cache=cache,
            return_cache=True,
        )
        assert next_cache is not None
        return next_cache

    @torch.no_grad()
    def generate(self, generation_config: GenerationConfig, **kwargs) -> GenerationResult:
        del kwargs
        config = generation_config
        if config.length % self.block_size:
            raise ValueError("generation length must be divisible by block_size")
        steps = (
            config.steps_per_block
            if config.steps_per_block is not None
            else config.num_steps
        )
        if steps <= 0:
            raise ValueError("steps_per_block must be positive")
        device = next(self.parameters()).device
        generator = seeded_generator(device, config.seed)
        prefix = torch.empty((config.batch_size, 0), device=device, dtype=torch.long)
        cache: list[KVCache] | None = None
        all_mask_counts: list[list[int]] = []
        all_entropies: list[list[float]] = []
        all_first_hitting_times: list[list[list[float]]] = []
        started = time.perf_counter()
        nfe = 0
        for _ in range(config.length // self.block_size):
            active = torch.full(
                (config.batch_size, self.block_size),
                self.mask_token_id,
                device=device,
                dtype=torch.long,
            )
            if self.ignore_bos and prefix.shape[1] == 0:
                if not 0 <= config.start_token_id < self.data_vocab_size:
                    raise ValueError("start_token_id must be a data-vocabulary BOS token")
                active[:, 0] = config.start_token_id
            mask_counts = [int(active.eq(self.mask_token_id).sum().item())]
            entropies: list[float] = []
            first_hitting_times: list[list[float]] = []
            if config.first_hitting:
                # Algorithm 1 in the official BD3-LM implementation: draw the
                # next first-hitting time and reveal one uniformly chosen masked
                # position in each sequence. A regular block completes after B
                # NFEs; the first TinyStories block needs B-1 because BOS is fixed.
                time_values = torch.ones(config.batch_size, device=device)
                for _ in range(steps):
                    masked = active.eq(self.mask_token_id)
                    masked_counts = masked.sum(-1)
                    if not masked.any():
                        break
                    uniform = torch.rand(
                        (config.batch_size,), device=device, generator=generator
                    ).clamp_min(torch.finfo(torch.float32).tiny)
                    time_values = time_values * uniform.pow(
                        masked_counts.clamp_min(1).to(torch.float32).reciprocal()
                    )
                    first_hitting_times.append(time_values.cpu().tolist())
                    logits = self._active_block_logits(prefix, active, time_values, cache)
                    proposals, probabilities = sample_logits_gumbel64(
                        logits, config, generator
                    )
                    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
                    entropies.append(float(entropy[masked].mean().item()))
                    priorities = torch.rand(active.shape, device=device, generator=generator)
                    priorities.masked_fill_(~masked, torch.inf)
                    chosen = priorities.argmin(-1, keepdim=True)
                    reveal = torch.zeros_like(masked).scatter_(1, chosen, True) & masked
                    active = torch.where(reveal, proposals, active)
                    mask_counts.append(int(active.eq(self.mask_token_id).sum().item()))
                    nfe += 1
            else:
                schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
                for t_tensor, s_tensor in zip(schedule[:-1], schedule[1:]):
                    t, s = float(t_tensor.item()), float(s_tensor.item())
                    logits = self._active_block_logits(prefix, active, t, cache)
                    proposals, probabilities = sample_logits_gumbel64(
                        logits, config, generator
                    )
                    masked = active.eq(self.mask_token_id)
                    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
                    entropies.append(float(entropy[masked].mean().item()) if masked.any() else 0.0)
                    reveal_probability = 1.0 if s == 0 else (t - s) / t
                    reveal = (
                        torch.rand(active.shape, device=device, generator=generator)
                        < reveal_probability
                    )
                    active = torch.where(masked & reveal, proposals, active)
                    mask_counts.append(int(active.eq(self.mask_token_id).sum().item()))
                    nfe += 1
            if active.eq(self.mask_token_id).any():
                if config.first_hitting:
                    raise RuntimeError(
                        "first-hitting sampling requires steps_per_block >= block_size"
                    )
                raise RuntimeError("exact final block transition left masked tokens")
            prefix_length = prefix.shape[1]
            prefix = torch.cat((prefix, active), dim=1)
            if config.use_kv_cache:
                cache = self._append_cache(active, prefix_length, cache)
            all_mask_counts.append(mask_counts)
            all_entropies.append(entropies)
            all_first_hitting_times.append(first_hitting_times)
        return GenerationResult(
            tokens=prefix,
            diagnostics={
                "num_function_evaluations": nfe,
                "steps_per_block": steps,
                "masked_tokens": all_mask_counts,
                "token_entropy": all_entropies,
                "sampler": "first_hitting" if config.first_hitting else "ancestral",
                "first_hitting_times": all_first_hitting_times,
                "wall_clock_seconds": time.perf_counter() - started,
                "kv_cache": config.use_kv_cache,
                "seed": config.seed,
                "one_step_is_exact_final_transition": not config.first_hitting and steps == 1,
            },
        )
