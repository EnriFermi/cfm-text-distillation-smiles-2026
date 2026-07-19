"""Tests for training-time BCFM (M3): mask, block DiT, ECLD jvp, training step, sampler.

All run on CPU with tiny dims (BlockDIT is pure torch — no flash-attn), so M3 is
verifiable end-to-end here, unlike M1/M2.
"""

import torch
import torch.nn.functional as F

from block.block_dit import BlockDIT
from block.block_semicat import BlockSemicatModule
from block.mask import block_causal_mask
from block.sampling import block_causal_sample

L, B, K = 8, 2, 5          # length, block_size, vocab -> 4 blocks, doubled length 16
NB = L // B


def _blk(pos):  # block id within a stream
    return (pos % L) // B


# --------------------------------------------------------------------------- mask
def test_mask_shape():
    m = block_causal_mask(L, B)
    assert m.shape == (2 * L, 2 * L) and m.dtype == torch.bool


def test_noisy_never_sees_own_or_future_clean():
    m = block_causal_mask(L, B)
    for qb in range(NB):
        q = L + qb * B                      # a noisy query in block qb
        for kb in range(NB):
            k = kb * B                       # a clean key in block kb
            assert m[q, k].item() == (kb < qb), (qb, kb)


def test_noisy_intra_block_only():
    m = block_causal_mask(L, B)
    for qb in range(NB):
        q = L + qb * B
        for kb in range(NB):
            k = L + kb * B                   # noisy key
            assert m[q, k].item() == (kb == qb)


def test_clean_is_block_causal_and_ignores_noisy():
    m = block_causal_mask(L, B)
    for qb in range(NB):
        q = qb * B
        for kb in range(NB):
            assert m[q, kb * B].item() == (kb <= qb)        # clean->clean block-causal
            assert m[q, L + kb * B].item() is False         # clean never sees noisy


def test_block0_noisy_has_only_itself():
    m = block_causal_mask(L, B)
    q = L                                    # block-0 noisy query
    allowed = m[q].nonzero().flatten().tolist()
    assert all(idx >= L and _blk(idx) == 0 for idx in allowed)


# ----------------------------------------------------------------------- block net
def _tiny_net():
    return BlockDIT(vocab_size=K, hidden_size=16, cond_dim=8, n_blocks=2,
                    n_heads=2, dropout=0.0, length=L, block_size=B)


def test_blockdit_forward_shape():
    net = _tiny_net().eval()
    x = torch.randn(3, 2 * L, K)
    s = torch.rand(3, 2 * L)
    t = torch.rand(3, 2 * L)
    out = net(x, s, t, block_causal_mask(L, B))
    assert out.shape == (3, 2 * L, K) and torch.isfinite(out).all()


def test_ecld_jvp_runs():
    net = _tiny_net().eval()
    mask = block_causal_mask(L, B)
    x = torch.randn(2, 2 * L, K)
    s = torch.rand(2, 2 * L)
    t = torch.rand(2, 2 * L)
    q, dq = torch.func.jvp(lambda _t: net(x, s, _t, mask).softmax(-1),
                           (t,), (torch.ones_like(t),))
    assert q.shape == dq.shape == (2, 2 * L, K)
    assert torch.isfinite(q).all() and torch.isfinite(dq).all()


def _dezero(net):
    """DiT uses adaLN-zero init, so an untrained net is the constant-0 map. Fill the
    zero-initialized params with noise so the network computes a non-trivial function
    and mask isolation becomes observable."""
    with torch.no_grad():
        for p in net.parameters():
            if float(p.abs().sum()) == 0.0:
                p.normal_(std=0.3)
    return net


def test_mask_isolation_in_real_net():
    """The correctness-critical property, checked through the actual network:
    a noisy block reacts to PREVIOUS clean blocks and is invariant to FUTURE/own ones."""
    net = _dezero(_tiny_net()).eval()
    mask = block_causal_mask(L, B)
    torch.manual_seed(0)
    clean = F.one_hot(torch.randint(0, K, (1, L)), K).float()
    noisy = torch.randn(1, L, K)
    s = torch.cat([torch.ones(1, L), torch.full((1, L), 0.3)], 1)
    t = torch.cat([torch.ones(1, L), torch.full((1, L), 0.6)], 1)

    def noisy_out(clean_stream):
        return net(torch.cat([clean_stream, noisy], 1), s, t, mask)[:, L:]

    base = noisy_out(clean)
    probe_block = 2                                # look at noisy block 2 -> sees clean 0,1
    seg = slice(probe_block * B, (probe_block + 1) * B)

    # change a FUTURE clean block (block 3): noisy block 2 must be unchanged
    fut = clean.clone(); fut[:, 3 * B:4 * B] = F.one_hot((torch.randint(0, K, (1, B)) + 1) % K, K).float()
    assert torch.allclose(noisy_out(fut)[:, seg], base[:, seg], atol=1e-5)

    # change its OWN clean block (block 2): still unchanged (no leakage)
    own = clean.clone(); own[:, seg] = F.one_hot((torch.randint(0, K, (1, B)) + 1) % K, K).float()
    assert torch.allclose(noisy_out(own)[:, seg], base[:, seg], atol=1e-5)

    # change a PREVIOUS clean block (block 1): noisy block 2 MUST change
    prev = clean.clone(); prev[:, B:2 * B] = F.one_hot((torch.randint(0, K, (1, B)) + 2) % K, K).float()
    assert not torch.allclose(noisy_out(prev)[:, seg], base[:, seg], atol=1e-5)


# -------------------------------------------------------------------------- module
def _tiny_module():
    return BlockSemicatModule(
        net=_tiny_net(), optimizer=None, scheduler=None,
        in_shape=(L, K), prior_type="gaussian", sd_prop=0.5, block_size=B,
    )


def test_training_step_finite():
    m = _tiny_module()
    x1 = torch.randint(0, K, (4, L))
    vf, sd = m.model_step(x1)
    assert torch.isfinite(vf) and vf > 0
    assert sd is not None and torch.isfinite(sd)


def test_block_causal_sample():
    m = _tiny_module().eval()
    toks = block_causal_sample(m, B, steps_per_block=2, batch_size=3, length=L)
    assert toks.shape == (3, L) and toks.dtype == torch.long
    assert 0 <= int(toks.min()) and int(toks.max()) < K


def test_cached_matches_full_sample():
    """KV-cached M3 sampling must match the full 2L masked path bit-for-bit."""
    m = _dezero(_tiny_module()).eval()
    torch.manual_seed(0)
    full = block_causal_sample(m, B, steps_per_block=2, batch_size=4, length=L, use_kv_cache=False)
    torch.manual_seed(0)
    cached = block_causal_sample(m, B, steps_per_block=2, batch_size=4, length=L, use_kv_cache=True)
    assert torch.equal(full, cached)


def test_cached_block_logits_match_full_forward():
    """Single-block noisy logits from forward_block == sliced full masked forward."""
    from block.block_dit import empty_kv_cache
    from block.mask import block_causal_mask

    net = _dezero(_tiny_net()).eval()
    mask = block_causal_mask(L, B)
    torch.manual_seed(1)
    clean_tok = torch.randint(0, K, (2, L))
    # finalize first two blocks as clean prefix, denoise block 2
    lo, hi = 2 * B, 3 * B
    z_blk = torch.randn(2, B, K)
    s_val, t_val = 0.2, 0.7

    clean_oh = F.one_hot(clean_tok, K).float()
    noisy = torch.zeros(2, L, K)
    noisy[:, lo:hi] = z_blk
    x = torch.cat([clean_oh, noisy], 1)
    s_tok = torch.zeros(2, L)
    t_tok = torch.zeros(2, L)
    s_tok[:, lo:hi], t_tok[:, lo:hi] = s_val, t_val
    ones = torch.ones(2, L)
    full_logits = net(x, torch.cat([ones, s_tok], 1), torch.cat([ones, t_tok], 1), mask)[:, L + lo:L + hi]

    kv = empty_kv_cache(len(net.blocks))
    for b in range(2):
        a, c = b * B, (b + 1) * B
        kv = net.encode_clean(clean_oh[:, a:c], torch.arange(a, c), kv)
    s_b = torch.full((2, B), s_val)
    t_b = torch.full((2, B), t_val)
    cached_logits = net.forward_block(z_blk, s_b, t_b, torch.arange(lo, hi), kv)
    assert torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-4)


def test_cache_grows_with_blocks():
    from block.block_dit import cache_seq_len, empty_kv_cache

    net = _tiny_net().eval()
    kv = empty_kv_cache(len(net.blocks))
    assert cache_seq_len(kv) == 0
    for b in range(NB):
        lo, hi = b * B, (b + 1) * B
        x = F.one_hot(torch.randint(0, K, (1, B)), K).float()
        kv = net.encode_clean(x, torch.arange(lo, hi), kv)
        assert cache_seq_len(kv) == hi


def test_sample_flow_map_batch_onehot():
    m = _tiny_module().eval()
    out = m.sample_flow_map_batch(batch_size=2, sampling_steps=1)
    assert out.shape == (2, L, K)
    assert torch.equal(out.sum(-1), torch.ones(2, L))  # valid one-hots
