"""Single source of truth for NFE accounting.

One NFE = one forward pass of the denoiser network. We report **NFE per token**,
the metric used in the proposal's speed axis.

- Full-sequence CFM produces all ``length`` tokens with ``steps`` forwards, so it
  is exactly the block case with ``block_size == length``.
- Blockwise BCFM does ``length / block_size`` blocks, each in ``steps_per_block``
  forwards.

How NFE relates to actual FLOPs differs between the two adaptations:

- **M2 (inference-time wrapper)** re-runs the *full-length* model every jump, so each
  of its NFEs costs the same as a full-sequence CFM NFE — NFE/token is directly
  FLOP-comparable between M1 and M2. Note this makes M2 *expensive*: B=16 with 2
  steps/block is 32 full forwards vs. 4 for a 4-step full-sequence CFM.
- **M3 (block-causal, KV-cached)** only processes the current block per forward, so a
  raw NFE count *overstates* its cost relative to M1/M2. ``forward_token_cost`` gives
  the attention-aware view for that case.

Wall-clock tokens/sec (recorded by the eval harness) is the ground truth; these
functions are the analytical axes for the paper's figures.
"""

from __future__ import annotations


def total_forwards(block_size: int, steps_per_block: int, length: int) -> int:
    if length % block_size != 0:
        raise ValueError(f"length {length} not divisible by block_size {block_size}")
    return (length // block_size) * steps_per_block


def nfe_per_token(block_size: int, steps_per_block: int, length: int) -> float:
    """The headline speed metric. Full-sequence CFM = ``block_size=length``."""
    return total_forwards(block_size, steps_per_block, length) / length


def flop_cost_per_token_full_recompute(block_size: int, steps_per_block: int, length: int) -> float:
    """Token-passes per generated token when every forward runs the full length.

    Applies to M1 (trivially: = steps) and M2 (the training-free wrapper): cost of one
    forward = ``length`` token-passes, so per generated token the cost equals the
    total number of forwards.
    """
    return float(total_forwards(block_size, steps_per_block, length))


def forward_token_cost(block_size: int, steps_per_block: int, length: int) -> float:
    """Token-passes per generated token for a KV-cached block-causal model (M3).

    Each block forward processes its own ``block_size`` tokens while attending to the
    cached prefix, so block ``b`` touches ``(b+1) * block_size`` token positions.
    """
    n_blocks = length // block_size
    seen = sum((b + 1) * block_size for b in range(n_blocks)) * steps_per_block
    return seen / length
