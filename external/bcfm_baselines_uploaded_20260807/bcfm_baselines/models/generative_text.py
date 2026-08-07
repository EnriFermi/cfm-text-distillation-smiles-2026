"""Model-neutral interfaces and sampling utilities for text generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import torch
from torch import Tensor, nn


@dataclass
class GenerationConfig:
    length: int = 256
    batch_size: int = 1
    strategy: Literal["sample", "greedy"] = "sample"
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_p_include_boundary: bool = True
    seed: int = 12345
    num_steps: int = 64
    steps_per_block: Optional[int] = None
    first_hitting: bool = False
    sampling_epsilon: float = 1e-5
    noise_removal: bool = True
    cache_denoiser_outputs: bool = True
    use_kv_cache: bool = True
    start_token_id: int = 27

    def __post_init__(self) -> None:
        if self.length <= 0 or self.batch_size <= 0:
            raise ValueError("length and batch_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.start_token_id < 0:
            raise ValueError("start_token_id must be non-negative")
        if not 0 <= self.sampling_epsilon < 1:
            raise ValueError("sampling_epsilon must be in [0, 1)")


@dataclass
class GenerationResult:
    tokens: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens.tolist(), "diagnostics": self.diagnostics}


class GenerativeTextModel(nn.Module, ABC):
    model_type: str

    @abstractmethod
    def compute_loss(self, batch: Tensor | dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return at least a scalar ``loss`` in nats per dataset token."""

    @abstractmethod
    @torch.no_grad()
    def generate(self, generation_config: GenerationConfig, **kwargs) -> GenerationResult:
        """Generate token IDs and return method-specific diagnostics."""

    def num_parameters(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable}

    def experiment_metadata(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "parameters": self.num_parameters()}


def batch_tokens(batch: Tensor | dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    if isinstance(batch, dict):
        tokens = batch["input_ids"]
        mask = batch.get("attention_mask", torch.ones_like(tokens, dtype=torch.bool))
    else:
        tokens = batch
        mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens.long(), mask.bool()


def seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def sample_time_values(
    count: int,
    device: torch.device,
    *,
    minimum: float,
    maximum: float = 1.0,
    stratified: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample diffusion times using the MDLM/BD3-LM stratified estimator.

    With ``stratified=True`` every element is sampled from a distinct equal-width
    partition of the unit interval, matching the official implementations.  The
    affine transform is deliberately applied after stratification rather than by
    clamping, so the result is uniform on ``[minimum, maximum]`` without endpoint
    point masses.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("time bounds must satisfy 0 <= minimum <= maximum <= 1")
    uniform = torch.rand(count, device=device, generator=generator)
    if stratified:
        offsets = torch.arange(count, device=device, dtype=uniform.dtype) / count
        uniform = (uniform / count + offsets) % 1.0
    return minimum + (maximum - minimum) * uniform


def filter_logits(
    logits: Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    *,
    include_top_p_boundary: bool = True,
) -> Tensor:
    logits = logits.clone()
    if top_k is not None and top_k < logits.shape[-1]:
        threshold = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < threshold, -torch.inf)
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        if include_top_p_boundary:
            remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits.masked_fill_(remove, -torch.inf)
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits


def sample_logits(
    logits: Tensor,
    config: GenerationConfig,
    generator: torch.Generator,
    *,
    include_top_p_boundary: bool | None = None,
) -> tuple[Tensor, Tensor]:
    if include_top_p_boundary is None:
        include_top_p_boundary = config.top_p_include_boundary
    logits = filter_logits(
        logits / config.temperature,
        config.top_k,
        config.top_p,
        include_top_p_boundary=include_top_p_boundary,
    )
    probabilities = logits.softmax(dim=-1)
    if config.strategy == "greedy":
        samples = probabilities.argmax(dim=-1)
    else:
        samples = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            1,
            generator=generator,
        ).reshape(probabilities.shape[:-1])
    return samples, probabilities
