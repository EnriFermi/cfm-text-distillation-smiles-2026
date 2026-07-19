from block.nfe import (
    cache_encode_forwards,
    context_token_cost_per_token,
    flow_forwards,
    full_sequence_token_cost,
    masked_2l_token_cost,
    nfe_per_token,
    network_forwards,
)

L = 256


def test_full_sequence_is_single_block():
    for steps in (1, 2, 4, 8, 16):
        assert flow_forwards(L, steps, L) == steps
        assert cache_encode_forwards(L, L) == 0
        assert network_forwards(L, steps, L, uses_kv_cache=False) == steps
        assert nfe_per_token(L, steps, L, uses_kv_cache=False) == steps / L


def test_cached_blockwise_counts_all_network_forwards():
    # B=16 has 16 denoise blocks and 15 useful clean-prefix cache builds.
    assert flow_forwards(16, 1, L) == 16
    assert cache_encode_forwards(16, L) == 15
    assert network_forwards(16, 1, L, uses_kv_cache=True) == 31
    assert nfe_per_token(16, 1, L, uses_kv_cache=True) == 31 / L


def test_full_sequence_token_cost_matches_flow_forwards():
    assert full_sequence_token_cost(L, 4, L) == 4.0
    assert full_sequence_token_cost(16, 2, L) == 32.0


def test_masked_reference_cost_doubles_sequence_length():
    assert masked_2l_token_cost(16, 2, L) == 64.0
    assert masked_2l_token_cost(L, 4, L) == 8.0


def test_cached_context_cost_includes_nonfinal_cache_builds():
    # 2 denoise steps: 2 * (1 + ... + 16) plus cache builds for blocks 0..14.
    assert context_token_cost_per_token(16, 2, L) == 24.5
