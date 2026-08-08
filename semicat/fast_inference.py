"""Inference-only compiled sampler for full-sequence categorical flow maps."""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch import Tensor, nn


class FullCFMFlowStep(nn.Module):
    """One fixed-shape Euler flow-map step, including softmax and state update."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, state: Tensor, s: Tensor, t: Tensor, step_scale: Tensor) -> Tensor:
        probabilities = self.net(state, s, t).softmax(dim=-1)
        return state + step_scale * (probabilities - state)


_COMPILED_FLOW_STEPS: dict[tuple[int, str], nn.Module] = {}


def _compile_mode() -> str:
    mode = os.environ.get("BCFM_INFERENCE_COMPILE_MODE", "reduce-overhead")
    if mode not in {"reduce-overhead", "max-autotune"}:
        raise ValueError(f"unsupported BCFM_INFERENCE_COMPILE_MODE={mode!r}")
    return mode


def compiled_full_cfm_flow_step(net: nn.Module) -> nn.Module:
    mode = _compile_mode()
    key = (id(net), mode)
    scorer = _COMPILED_FLOW_STEPS.get(key)
    if scorer is None:
        scorer = torch.compile(
            FullCFMFlowStep(net),
            fullgraph=True,
            mode=mode,
        )
        _COMPILED_FLOW_STEPS[key] = scorer
    return scorer


@torch.inference_mode()
def full_cfm_sample_fused_fast(
    module,
    sampling_steps: int,
    *,
    batch_size: int,
    discretize: Literal["argmax", "sample"] = "argmax",
    compile_step: bool = True,
) -> Tensor:
    """Generate tokens with the full CFM while keeping every Euler step compiled."""
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    length, vocab = (int(value) for value in module.in_shape)
    device = module.device
    state = module.prior((batch_size, length, vocab), device=device)
    scorer: nn.Module = (
        compiled_full_cfm_flow_step(module.net)
        if compile_step else FullCFMFlowStep(module.net)
    )
    schedule = torch.linspace(
        0.0, 1.0, sampling_steps + 1, device=device, dtype=state.dtype,
    )
    for s_value, t_value in zip(schedule[:-1], schedule[1:]):
        s = s_value.expand(batch_size)
        t = t_value.expand(batch_size)
        step_scale = (t_value - s_value) / (1.0 - s_value + 1e-8)
        state = scorer(state, s, t, step_scale)
        if compile_step:
            # Inductor CUDA graphs own reusable output storage.  Preserve the state
            # consumed by the next replay.
            state = state.clone()
    if discretize == "argmax":
        return state.argmax(dim=-1)
    if discretize == "sample":
        probabilities = state.clamp_min(0).flatten(0, 1)
        return torch.multinomial(probabilities, 1).squeeze(-1).view(batch_size, length)
    raise ValueError(f"unsupported discretize={discretize!r}")
