"""Sample-based generative evaluation (Gen-PPL and MAUVE) for every baseline.

Both the in-training validation loop and the offline evaluator go through this
module, so a checkpoint's Gen-PPL curve during training and its final reported
number are produced by identical code -- only the sampling budget differs.

Generation is chunked: the sample count is usually far larger than a batch that
fits alongside a training step, and every chunk gets its own derived seed so the
chunks are not identical draws (each model's ``generate`` reseeds from
``GenerationConfig.seed``).  The human reference for MAUVE is drawn from the
validation split with a fixed generator, which gives independent windows rather
than the stride-1 overlapping ones ``sequential_batches`` produces.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import time
from typing import Any

import torch
from torch import Tensor

from bcfm_baselines.metrics import DEFAULT_MAUVE_MODEL, DEFAULT_PPL_MODEL, TextMetrics
from bcfm_baselines.models.generative_text import GenerationConfig


@dataclass
class GenerativeEvalConfig:
    """Resolved ``generative_evaluation`` block."""

    enabled: bool = True
    sample_count: int = 256
    sampling_batch_size: int = 64
    seed: int = 12345
    num_steps: int | None = None
    steps_per_block: int | None = None
    first_hitting: bool | None = None
    ppl_model: str = DEFAULT_PPL_MODEL
    ppl_batch_size: int = 32
    ppl_context_size: int | None = None
    mauve_model: str = DEFAULT_MAUVE_MODEL
    mauve_reference_count: int | None = None
    mauve_scaling_factor: float = 5.0
    mauve_seed: int = 12345
    mauve_batch_size: int = 16
    reference_split: str = "validation"

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.sampling_batch_size <= 0:
            raise ValueError("sample_count and sampling_batch_size must be positive")
        if self.ppl_batch_size <= 0:
            raise ValueError("ppl_batch_size must be positive")


def require_metric_dependencies(settings: GenerativeEvalConfig) -> None:
    """Fail before training when enabled Gen-PPL/MAUVE packages are absent."""
    if not settings.enabled:
        return
    modules = {
        "transformers": "transformers==4.41.0",
        "mauve": "mauve-text>=0.4.0",
        "sklearn": "scikit-learn>=1.3",
        "faiss": "faiss-cpu>=1.7.4",
    }
    failures = []
    for module, requirement in modules.items():
        try:
            importlib.import_module(module)
        except Exception as error:
            failures.append(f"{requirement} ({type(error).__name__}: {error})")
    if failures:
        raise RuntimeError(
            "generative evaluation is enabled but required packages are missing: "
            f"{', '.join(failures)}. Install requirements-a100.txt in the Python "
            "environment used by the job before starting training."
        )


def resolve_config(config: dict[str, Any]) -> GenerativeEvalConfig:
    """Build a :class:`GenerativeEvalConfig`, tolerating configs without the block."""
    raw = dict(config.get("generative_evaluation") or {})
    known = GenerativeEvalConfig.__dataclass_fields__
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown generative_evaluation keys: {sorted(unknown)}")
    return GenerativeEvalConfig(**raw)


def _generation_config(
    sampling: dict[str, Any],
    settings: GenerativeEvalConfig,
    batch_size: int,
    seed: int,
) -> GenerationConfig:
    values = dict(sampling)
    values["batch_size"] = batch_size
    values["seed"] = seed
    if settings.num_steps is not None:
        values["num_steps"] = settings.num_steps
    # Only override a block schedule the model actually has one for; AR and MDLM
    # keep steps_per_block at null.
    if settings.steps_per_block is not None and values.get("steps_per_block") is not None:
        values["steps_per_block"] = settings.steps_per_block
    if settings.first_hitting is not None and "first_hitting" in values:
        values["first_hitting"] = settings.first_hitting
    return GenerationConfig(**values)


@torch.no_grad()
def sample_texts(
    model,
    corpus,
    sampling: dict[str, Any],
    settings: GenerativeEvalConfig,
) -> tuple[list[str], list[Tensor], dict[str, Any]]:
    """Draw ``settings.sample_count`` samples in chunks and detokenize them."""
    texts: list[str] = []
    token_batches: list[Tensor] = []
    diagnostics: dict[str, Any] = {}
    remaining = settings.sample_count
    chunk_index = 0
    started = time.perf_counter()
    while remaining > 0:
        size = min(settings.sampling_batch_size, remaining)
        generation = _generation_config(
            sampling, settings, size, settings.seed + chunk_index
        )
        result = model.generate(generation)
        tokens = result.tokens.detach().cpu()
        token_batches.append(tokens)
        texts.extend(corpus.decode(tokens))
        if chunk_index == 0:
            diagnostics = dict(result.diagnostics)
        remaining -= size
        chunk_index += 1
    diagnostics["generation_seconds"] = time.perf_counter() - started
    diagnostics["generation_chunks"] = chunk_index
    return texts, token_batches, diagnostics


def reference_texts(corpus, count: int, seed: int, split: str = "validation") -> list[str]:
    """Detokenized human windows from ``split``, drawn independently at random."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    collected: list[str] = []
    remaining = count
    while remaining > 0:
        size = min(256, remaining)
        batch = corpus.random_batch(split, size, generator)
        collected.extend(corpus.decode(batch))
        remaining -= size
    return collected


def evaluate_generation(
    model,
    corpus,
    sampling: dict[str, Any],
    settings: GenerativeEvalConfig,
    device: torch.device | str,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Sample, then score with entropy, Gen-PPL, and MAUVE.

    Returns an empty dict when generative evaluation is switched off.
    """
    if not settings.enabled:
        return {}
    require_metric_dependencies(settings)
    was_training = model.training
    model.eval()
    try:
        texts, token_batches, diagnostics = sample_texts(model, corpus, sampling, settings)
        context_size = settings.ppl_context_size or corpus.sequence_length
        references = reference_texts(
            corpus,
            settings.mauve_reference_count or settings.sample_count,
            settings.mauve_seed,
            settings.reference_split,
        )
        metrics = TextMetrics.compute_all(
            texts,
            references,
            token_batches,
            context_size=context_size,
            ppl_model=settings.ppl_model,
            ppl_batch_size=settings.ppl_batch_size,
            mauve_model=settings.mauve_model,
            mauve_scaling_factor=settings.mauve_scaling_factor,
            mauve_seed=settings.mauve_seed,
            mauve_batch_size=settings.mauve_batch_size,
            device=str(device),
        )
    finally:
        if was_training:
            model.train()
    metrics["generation_seconds"] = diagnostics.get("generation_seconds")
    for key in ("num_function_evaluations", "steps_per_block", "sampler"):
        if key in diagnostics:
            metrics[f"generation_{key}"] = diagnostics[key]
    metrics["generation_first_hitting"] = diagnostics.get("sampler") == "first_hitting"
    metrics["sample_texts"] = texts[:16]
    return {f"{prefix}{key}": value for key, value in metrics.items()}
