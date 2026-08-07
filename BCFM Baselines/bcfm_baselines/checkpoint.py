from __future__ import annotations

from pathlib import Path

import torch

from bcfm_baselines.factory import create_model


def load_model_state(model, state: dict[str, torch.Tensor]) -> None:
    """Load model weights, allowing only known additions to legacy checkpoints."""
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = (
        {"mask_rate_min", "mask_rate_max"}
        if getattr(model, "model_type", None) == "bd3lm"
        else set()
    )
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint is incompatible with the configured model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def load_checkpoint_model(path: str | Path, device: torch.device | str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = create_model(payload["config"])
    state = payload.get("ema_model") or payload["model"]
    load_model_state(model, state)
    model.to(device).eval()
    payload["weights_used_for_inference"] = "ema" if payload.get("ema_model") else "online"
    return model, payload
