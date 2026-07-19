"""Behavioral tests for the KV-cached blockwise sampler.

FakeCFM mimics the sampler interface without data or a trained model.
"""

import torch

from block.sampling import blockwise_sample

K = 27


class FakeCFM:
    def __init__(self, length: int, k: int = K):
        self.in_shape = (length, k)
        self.device = "cpu"
        g = torch.Generator().manual_seed(7)
        self.W = torch.randn(k, k, generator=g)
        self.V = torch.randn(k, k, generator=g)  # cross-position mixing ("attention")
        self.net = FakeCachedNet(k)

    def prior(self, shape, device):
        return torch.randn(shape, device=device)

    def xst(self, x, s, t):
        dt = ((t - s) / (1.0 - s + 1e-8)).view(-1, 1, 1)
        # mean-pooled term makes every position depend on the whole sequence, so
        # prefix content can influence the current block (like attention would)
        mixed = x.mean(dim=1, keepdim=True) @ self.V
        q = (x @ self.W + mixed + s.view(-1, 1, 1) + t.view(-1, 1, 1)).softmax(-1)
        return x + dt * (q - x)


def test_shapes_and_range():
    m = FakeCFM(32)
    toks = blockwise_sample(m, block_size=8, steps_per_block=2, batch_size=5, length=32)
    assert toks.shape == (5, 32) and toks.dtype == torch.long
    assert 0 <= int(toks.min()) and int(toks.max()) < K


def test_gold_prefix_conditions_but_returns_generated():
    """E6 semantics: gold teacher-forces the *conditioning*, output stays generated."""
    L, bs = 16, 4
    m = FakeCFM(L)
    gold_a = torch.zeros(bs, L, dtype=torch.long)
    gold_b = torch.full((bs, L), K - 1, dtype=torch.long)

    torch.manual_seed(0)
    out_a = blockwise_sample(m, 4, 1, batch_size=bs, length=L, gold_prefix=gold_a)
    torch.manual_seed(0)
    out_b = blockwise_sample(m, 4, 1, batch_size=bs, length=L, gold_prefix=gold_b)

    # output is the model's, not the gold tokens
    assert not torch.equal(out_a, gold_a)
    # block 1 has no prefix: identical across golds; later blocks see different
    # conditioning and must diverge
    assert torch.equal(out_a[:, :4], out_b[:, :4])
    assert not torch.equal(out_a[:, 4:], out_b[:, 4:])


def test_generated_prefix_actually_conditions():
    """Different prior draws in block 1 must propagate into later blocks."""
    L, bs = 16, 4
    m = FakeCFM(L)
    torch.manual_seed(0)
    a = blockwise_sample(m, 4, 2, batch_size=bs, length=L)
    torch.manual_seed(1)
    b = blockwise_sample(m, 4, 2, batch_size=bs, length=L)
    assert not torch.equal(a, b)


def test_sample_discretize_runs():
    m = FakeCFM(16)
    toks = blockwise_sample(m, 4, 2, batch_size=3, length=16, discretize="sample")
    assert toks.shape == (3, 16)
    assert 0 <= int(toks.min()) and int(toks.max()) < K


class FakeCachedNet:
    """Minimal net with encode_clean / forward_block for M2 KV-cache tests."""

    def __init__(self, k: int = K, n_layers: int = 2):
        self.blocks = list(range(n_layers))  # len() only
        g = torch.Generator().manual_seed(13)
        self.W = torch.randn(k, k, generator=g)
        self.U = torch.randn(k, k, generator=g)
        self.encode_clean_calls = 0

    def encode_clean(self, x_clean, positions, kv_cache=None):
        self.encode_clean_calls += 1
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        # K/V stand-in: (batch, B, K) — append to prefix so cache_seq_len grows
        new_cache = []
        for entry in kv_cache:
            if entry is None:
                new_cache.append((x_clean, x_clean))
            else:
                k_c, v_c = entry
                new_cache.append((torch.cat([k_c, x_clean], dim=1),
                                  torch.cat([v_c, x_clean], dim=1)))
        return new_cache

    def forward_block(self, x_noisy, s, t, positions, kv_cache=None):
        ctx = torch.zeros(x_noisy.shape[0], 1, x_noisy.shape[-1], device=x_noisy.device)
        if kv_cache is not None and kv_cache[0] is not None:
            ctx = kv_cache[0][0].mean(dim=1, keepdim=True)
        return x_noisy @ self.W + ctx @ self.U + s.view(-1, 1, 1)


class FakeCachedCFM(FakeCFM):
    def __init__(self, length: int, k: int = K):
        super().__init__(length, k)
        self.net = FakeCachedNet(k)


def test_cached_sample_shapes_and_flexible_length():
    from block.sampling import blockwise_sample
    m = FakeCachedCFM(256)
    toks = blockwise_sample(
        m, block_size=8, steps_per_block=2, batch_size=3, num_blocks=4,
    )
    assert toks.shape == (3, 32) and toks.dtype == torch.long
    assert 0 <= int(toks.min()) and int(toks.max()) < K
    assert m.net.encode_clean_calls == 3  # final block cache is never consumed


def test_cached_prefix_conditions_later_blocks():
    from block.sampling import blockwise_sample
    m = FakeCachedCFM(64)
    gold_a = torch.zeros(2, 16, dtype=torch.long)
    gold_b = torch.full((2, 16), K - 1, dtype=torch.long)
    torch.manual_seed(0)
    out_a = blockwise_sample(
        m, 4, 1, batch_size=2, num_blocks=4, gold_prefix=gold_a,
    )
    torch.manual_seed(0)
    out_b = blockwise_sample(
        m, 4, 1, batch_size=2, num_blocks=4, gold_prefix=gold_b,
    )
    assert torch.equal(out_a[:, :4], out_b[:, :4])
    assert not torch.equal(out_a[:, 4:], out_b[:, 4:])


def _dezero(net):
    """Make the adaLN-zero initialization observable in a tiny inference test."""
    with torch.no_grad():
        for parameter in net.parameters():
            if float(parameter.abs().sum()) == 0.0:
                parameter.normal_(std=0.2)
    return net


def test_duo_dit_cache_appends_clean_prefix_and_is_read_only_for_denoising():
    """Exercise M2's production DIT cache API, not only the fake sampler net."""
    from semicat.net.duo import DIT, cache_seq_len, empty_kv_cache

    vocab, block_size = 5, 2
    net = _dezero(DIT(
        vocab_size=vocab, hidden_size=16, cond_dim=8, n_blocks=2, n_heads=2,
        dropout=0.0, length=4,
    )).eval()
    prefix_a = torch.nn.functional.one_hot(torch.tensor([[0, 1]]), vocab).float()
    prefix_b = torch.nn.functional.one_hot(torch.tensor([[2, 3]]), vocab).float()
    noisy = torch.randn(1, block_size, vocab)
    s = torch.full((1,), 0.25)
    t = torch.full((1,), 0.75)

    cache_a = net.encode_clean(prefix_a, torch.arange(0, block_size), empty_kv_cache(2))
    cache_b = net.encode_clean(prefix_b, torch.arange(0, block_size), empty_kv_cache(2))
    assert cache_seq_len(cache_a) == block_size

    cache_snapshot = [(k.clone(), v.clone()) for k, v in cache_a]
    logits_a = net.forward_block(noisy, s, t, torch.arange(2, 4), cache_a)
    logits_b = net.forward_block(noisy, s, t, torch.arange(2, 4), cache_b)
    assert not torch.allclose(logits_a, logits_b)
    for (k_before, v_before), (k_after, v_after) in zip(cache_snapshot, cache_a):
        assert torch.equal(k_before, k_after)
        assert torch.equal(v_before, v_after)

    cache_a = net.encode_clean(
        prefix_a, torch.arange(2, 4), cache_a,
    )
    assert cache_seq_len(cache_a) == 2 * block_size
