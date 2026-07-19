"""Single source of truth for inference work accounting.

An NFE is one invocation of transformer layers.  In cached blockwise inference this
includes both a denoising ``forward_block`` and an ``encode_clean`` invocation that
constructs the K/V prefix needed by a later block.  The latter is not a flow-map
evaluation, so it is also reported separately.

``context_token_cost_per_token`` is an analytical proxy rather than measured FLOPs:
each block pass is charged for its current block plus the clean K/V context it reads.
Use measured ``tokens_per_sec`` for hardware-level comparisons.
"""

from __future__ import annotations


def _n_blocks(block_size: int, length: int) -> int:
    if length % block_size != 0:
        raise ValueError(f"length {length} not divisible by block_size {block_size}")
    return length // block_size


def flow_forwards(block_size: int, steps_per_block: int, length: int) -> int:
    """Number of flow-map denoising calls, excluding cache construction."""
    return _n_blocks(block_size, length) * steps_per_block


def cache_encode_forwards(block_size: int, length: int) -> int:
    """Clean-prefix cache builds required by cached generation.

    The final generated block is deliberately not encoded because no later block can
    consume its K/V cache.
    """
    return max(_n_blocks(block_size, length) - 1, 0)


def network_forwards(
    block_size: int,
    steps_per_block: int,
    length: int,
    *,
    uses_kv_cache: bool,
) -> int:
    """All transformer-layer invocations performed while generating a sequence."""
    total = flow_forwards(block_size, steps_per_block, length)
    return total + (cache_encode_forwards(block_size, length) if uses_kv_cache else 0)


def nfe_per_token(
    block_size: int,
    steps_per_block: int,
    length: int,
    *,
    uses_kv_cache: bool,
) -> float:
    """All network forwards per generated token."""
    return network_forwards(
        block_size, steps_per_block, length, uses_kv_cache=uses_kv_cache,
    ) / length


def full_sequence_token_cost(
    block_size: int,
    steps_per_block: int,
    length: int,
) -> float:
    """Token-pass proxy when each flow evaluation processes all ``length`` tokens."""
    return float(flow_forwards(block_size, steps_per_block, length))


def masked_2l_token_cost(
    block_size: int,
    steps_per_block: int,
    length: int,
) -> float:
    """Token-pass proxy for the M3 doubled-stream reference sampler."""
    return 2.0 * full_sequence_token_cost(block_size, steps_per_block, length)


def context_token_cost_per_token(
    block_size: int,
    steps_per_block: int,
    length: int,
) -> float:
    """Context-token proxy for KV-cached blockwise inference.

    A denoising pass for block ``b`` reads ``(b + 1) * block_size`` positions: its
    current block and the cached prefix.  Each non-final block performs one additional
    clean-prefix cache build with the same context size.
    """
    n_blocks = _n_blocks(block_size, length)
    per_block_context = [(b + 1) * block_size for b in range(n_blocks)]
    flow_context = sum(per_block_context) * steps_per_block
    cache_context = sum(per_block_context[:-1])
    return (flow_context + cache_context) / length
