"""Blockwise (semi-autoregressive) sampling for BCFM — Algorithm 1 of the proposal.

Generate a sequence block by block: each block is produced in a few flow-map jumps
from a fresh simplex prior, conditioned on the finalized (discretized) prefix, then
appended to that prefix.

The same loop serves both adaptations:

- **Inference-time (M2)**: wraps a pretrained *full-sequence* CFM. The map is the
  existing ``SemicatModule.xst`` applied to the whole length. Prefix positions are
  pinned after every jump so the map cannot move them (see ``prefix_mode``); positions
  after the current block start as prior noise and evolve under the map within the
  block, then get re-drawn at the next block start.
- **Training-time (M3, later)**: pass a block-causal ``map_fn``; the rest is identical.

Because the full-sequence model has a single global time (s, t), a clean prefix at
small ``s`` is off the training distribution — that mismatch is exactly what the
inference-time adaptation probes ("how much semi-AR behavior is free"). ``prefix_mode``
selects the heuristic: ``clean`` feeds the one-hot prefix as-is (the proposal's literal
reading); ``renoise`` keeps the prefix on the training interpolant
``(1-u) z_0 + u one_hot`` at the current time ``u``, trading conditioning strength for
being on-distribution.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor
from torch.nn import functional as F


def uniform_schedule(steps: int) -> list[tuple[float, float]]:
    """``steps`` equal jumps covering [0, 1]: e.g. 1 -> [(0,1)], 2 -> [(0,.5),(.5,1)]."""
    grid = torch.linspace(0.0, 1.0, steps + 1).tolist()
    return list(zip(grid[:-1], grid[1:]))


# A map step: (state, s, t) -> new state, all full-length. Defaults to module.xst.
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
    """Sample a batch of token sequences block by block.

    :param module: a ``SemicatModule`` (provides ``.xst``, ``.prior``, ``.in_shape``,
        ``.device``).
    :param block_size: block length ``B``. ``block_size == length`` recovers the
        full-sequence sampler (single block) — used as a sanity check.
    :param steps_per_block: flow-map jumps per block (the ``1/2/4`` axis). Ignored if
        ``schedule`` is given.
    :param batch_size: number of sequences to sample.
    :param length: sequence length ``L``; defaults to ``module.in_shape[0]``.
    :param schedule: explicit in-block ``[(s, t), ...]`` grid; defaults to a uniform
        grid with ``steps_per_block`` jumps.
    :param discretize: how to read tokens off the block endpoint distribution.
    :param prefix_mode: how finalized blocks are presented to the model (see module
        docstring). ``renoise`` needs >1 steps/block to carry any prefix signal (at
        ``s=0`` a re-noised prefix is pure noise).
    :param gold_prefix: ``(batch, length)`` gold tokens. If given, each block is
        *conditioned* on the gold prefix (teacher forcing) while the **generated**
        blocks are still what is returned — the exposure-bias probe compares metrics
        of these blocks against normally-generated ones.
    :param map_fn: override the flow-map step (M3). Defaults to ``module.xst``.
    :return: ``(batch, length)`` long tensor of **generated** token ids.
    """
    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size != 0:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    step = map_fn or module.xst
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]

    z = module.prior((batch_size, L, K), device=device)
    z0_ref = z.clone()  # frozen prior draw, reused when re-noising the prefix
    generated = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    cond = torch.zeros_like(generated)  # conditioning stream: generated or gold tokens

    def pin_prefix(lo: int, u: float) -> None:
        """Reset positions < lo to the prefix representation at time ``u``."""
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
        # Fresh prior noise for the current block and everything after it.
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
        z[:, lo:hi] = F.one_hot(cond[:, lo:hi], K).to(z.dtype)  # finalize for conditioning

    return generated


@torch.inference_mode()
def block_causal_sample(
    module,
    block_size: int,
    steps_per_block: int,
    *,
    batch_size: int,
    length: int | None = None,
    schedule: list[tuple[float, float]] | None = None,
    discretize: Literal["argmax", "sample"] = "argmax",
    inference_backend: Literal[
        "full", "cached", "compiled_cached", "fused_cached", "compiled_fused_cached"
    ] = "full",
) -> Tensor:
    """Sampler for the trained block-causal model (M3).

    ``inference_backend='full'`` preserves the training-style doubled-sequence path.
    ``'cached'`` evaluates only the current block with an incremental clean-prefix KV
    cache, and ``'compiled_cached'`` additionally compiles the block flow step and
    clean encoder for maximum steady-state speed.  The cached graph is algebraically
    identical to the mask but may differ numerically on CUDA because it uses dense
    SDPA rather than FlexAttention's reduction order.
    """
    if inference_backend != "full":
        if inference_backend not in {
            "cached", "compiled_cached", "fused_cached", "compiled_fused_cached"
        }:
            raise ValueError(f"unknown inference_backend={inference_backend!r}")
        from block.fast_inference import (
            block_causal_sample_cached,
            block_causal_sample_fused_cached,
        )

        if inference_backend in {"fused_cached", "compiled_fused_cached"}:
            return block_causal_sample_fused_cached(
                module,
                block_size,
                steps_per_block,
                batch_size=batch_size,
                length=length,
                schedule=schedule,
                discretize=discretize,
                compile_steps=inference_backend == "compiled_fused_cached",
            )

        return block_causal_sample_cached(
            module,
            block_size,
            steps_per_block,
            batch_size=batch_size,
            length=length,
            schedule=schedule,
            discretize=discretize,
            compile_noisy=inference_backend == "compiled_cached",
            compile_strategy="dynamic",
        )

    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size != 0:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    sched = [(float(s), float(t)) for s, t in (schedule or uniform_schedule(steps_per_block))]

    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    for b in range(L // block_size):
        lo, hi = b * block_size, (b + 1) * block_size
        z_blk = module.prior((batch_size, block_size, K), device=device)
        for s, t in sched:
            clean = F.one_hot(tokens, K).to(z_blk.dtype)            # finalized prefix (rest masked)
            noisy = torch.zeros(batch_size, L, K, device=device, dtype=z_blk.dtype)
            noisy[:, lo:hi] = z_blk
            s_tok = torch.zeros(batch_size, L, device=device)
            t_tok = torch.zeros(batch_size, L, device=device)
            s_tok[:, lo:hi], t_tok[:, lo:hi] = s, t
            ones = torch.ones(batch_size, L, device=device)
            x = torch.cat([clean, noisy], dim=1)
            s_full = torch.cat([ones, s_tok], dim=1)
            t_full = torch.cat([ones, t_tok], dim=1)
            q_blk = module.net(x, s_full, t_full)[:, L + lo:L + hi].softmax(dim=-1)
            z_blk = z_blk + ((t - s) / (1.0 - s + 1e-8)) * (q_blk - z_blk)
        tokens[:, lo:hi] = _discretize(z_blk, discretize)
    return tokens


def _discretize(endpoint: Tensor, how: str) -> Tensor:
    """``endpoint`` is a categorical distribution on the simplex, ``(batch, B, K)``."""
    if how == "argmax":
        return endpoint.argmax(dim=-1)
    if how == "sample":
        probs = endpoint.clamp_min(0).flatten(0, 1)
        return torch.multinomial(probs, 1).squeeze(-1).view(endpoint.shape[:-1])
    raise ValueError(f"unknown discretize mode '{how}'")
