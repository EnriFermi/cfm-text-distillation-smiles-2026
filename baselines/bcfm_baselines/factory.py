"""Configuration-selected model factory."""

from __future__ import annotations

from typing import Any

from bcfm_baselines.models import (
    AutoregressiveTextModel,
    BlockDiffusionTextModel,
    MaskedDiffusionTextModel,
)


def create_model(config: dict[str, Any]):
    model_config = config["model"]
    model_type = model_config["type"].lower()
    common = {
        "backbone": model_config["backbone"],
        "data_vocab_size": model_config.get("data_vocab_size", 27),
        "mask_token_id": model_config.get("mask_token_id", 27),
    }
    if model_type == "ar":
        return AutoregressiveTextModel(
            **common,
            bos_token_id=model_config.get(
                "bos_token_id", model_config.get("mask_token_id", 27)
            ),
        )
    if model_type == "mdlm":
        return MaskedDiffusionTextModel(
            **common,
            min_time=model_config.get("min_time", 1e-3),
            antithetic_time_sampling=model_config.get("antithetic_time_sampling", True),
            time_conditioning=model_config.get("time_conditioning", False),
        )
    if model_type == "bd3lm":
        return BlockDiffusionTextModel(
            **common,
            block_size=model_config.get("block_size", 16),
            min_time=model_config.get("min_time", 1e-3),
            antithetic_time_sampling=model_config.get("antithetic_time_sampling", True),
            time_conditioning=model_config.get("time_conditioning", False),
            mask_rate_min=model_config.get("mask_rate_min", 1e-3),
            mask_rate_max=model_config.get("mask_rate_max", 1.0),
            schedule_search=model_config.get("schedule_search", True),
            schedule_search_widths=model_config.get(
                "schedule_search_widths", (0.5, 0.6, 0.7, 0.8, 0.9)
            ),
            schedule_search_delta=model_config.get("schedule_search_delta", 0.05),
            schedule_search_batches=model_config.get("schedule_search_batches", 100),
            resample_masked_fraction=model_config.get("resample_masked_fraction", True),
        )
    raise ValueError(f"unknown model.type {model_type!r}; expected ar, mdlm, or bd3lm")
