"""Pin-prefix blockwise sampling — legacy M2 heuristic, tests only.

Production inference uses ``block.sampling.blockwise_sample`` (BD3-LM mask).
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from block.sampling import _discretize, uniform_schedule

MapFn = Callable[[Tensor, Tensor, Tensor], Tensor]


@torch.inference_mode()
def blockwise_sample(
    module,
    block_size: int,
    steps_per_block: int,
    *,
    batch_size: int,
    length: int | None = None,
    schedule: list[tuple[float, float]] | None = None,
    discretize: Literal["argmax", "sample"] = "argmax",
    prefix_mode: Literal["clean", "renoise"] = "clean",
    gold_prefix: Tensor | None = None,
    map_fn: MapFn | None = None,
) -> Tensor:
    """Pin-prefix blockwise sampler (legacy). See ``block.sampling`` for the proper M2 path."""
    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size != 0:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    step = map_fn or module.xst
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]

    z = module.prior((batch_size, L, K), device=device)
    z0_ref = z.clone()
    generated = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    cond = torch.zeros_like(generated)

    def pin_prefix(lo: int, u: float) -> None:
        if lo == 0:
            return
        clean = F.one_hot(cond[:, :lo], K).to(z.dtype)
        if prefix_mode == "clean" or u >= 1.0:
            z[:, :lo] = clean
        elif prefix_mode == "renoise":
            z[:, :lo] = (1.0 - u) * z0_ref[:, :lo] + u * clean
        else:
            raise ValueError(f"unknown prefix_mode '{prefix_mode}'")

    for b in range(L // block_size):
        lo, hi = b * block_size, (b + 1) * block_size
        z[:, lo:] = module.prior((batch_size, L - lo, K), device=device)
        pin_prefix(lo, sched[0][0])

        for s, t in sched:
            s_b = torch.full((batch_size,), s, device=device)
            t_b = torch.full((batch_size,), t, device=device)
            z = step(z, s_b, t_b)
            pin_prefix(lo, t)

        block = _discretize(z[:, lo:hi], discretize)
        generated[:, lo:hi] = block
        cond[:, lo:hi] = gold_prefix[:, lo:hi].to(device) if gold_prefix is not None else block
        z[:, lo:hi] = F.one_hot(cond[:, lo:hi], K).to(z.dtype)

    return generated
