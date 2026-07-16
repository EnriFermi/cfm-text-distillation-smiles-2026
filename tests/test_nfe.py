from block.nfe import (
    flop_cost_per_token_full_recompute,
    flop_cost_per_token_masked,
    forward_token_cost,
    nfe_per_token,
    total_forwards,
)

L = 256


def test_full_sequence_is_single_block():
    # full-seq CFM with N steps == blockwise with block_size=L, steps_per_block=N
    for steps in (1, 2, 4, 8, 16):
        assert nfe_per_token(L, steps, L) == steps / L


def test_blockwise_counts():
    assert total_forwards(16, 2, L) == 32
    assert nfe_per_token(16, 2, L) == 0.125


def test_flop_cost_full_recompute_matches_forwards():
    # M1: cost per token == steps; pin-prefix M2: == total forwards
    assert flop_cost_per_token_full_recompute(L, 4, L) == 4.0
    assert flop_cost_per_token_full_recompute(16, 2, L) == 32.0


def test_flop_cost_masked_doubles_full_recompute():
    assert flop_cost_per_token_masked(16, 2, L) == 64.0
    assert flop_cost_per_token_masked(L, 4, L) == 8.0


def test_cached_cost_below_full_recompute():
    # KV-cached blockwise must be cheaper than recomputing the full sequence
    assert forward_token_cost(16, 2, L) < flop_cost_per_token_full_recompute(16, 2, L)
