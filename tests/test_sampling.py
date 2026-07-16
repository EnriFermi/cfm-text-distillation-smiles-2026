"""Behavioral tests for the blockwise sampler (no data / no real net needed).

FakeCFM mimics ``SemicatModule``'s sampling interface (``prior``, ``xst``,
``in_shape``, ``device``) with a fixed random linear "net", so trajectories are
deterministic given the torch seed and *depend on the input state* — enough to test
conditioning behavior, not just shapes.
"""

import torch

from block.sampling_pin import blockwise_sample
from block.sampling import uniform_schedule

K = 27


class FakeCFM:
    def __init__(self, length: int, k: int = K):
        self.in_shape = (length, k)
        self.device = "cpu"
        g = torch.Generator().manual_seed(7)
        self.W = torch.randn(k, k, generator=g)
        self.V = torch.randn(k, k, generator=g)  # cross-position mixing ("attention")

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


def test_single_block_reproduces_full_sequence_sampler():
    """block_size == L must equal the plain full-sequence few-step sampler bit-for-bit."""
    L, bs, steps = 16, 4, 3
    m = FakeCFM(L)
    torch.manual_seed(42)
    blockwise = blockwise_sample(m, block_size=L, steps_per_block=steps, batch_size=bs, length=L)

    torch.manual_seed(42)
    _ = m.prior((bs, L, K), "cpu")  # replicate blockwise's init draw
    z = m.prior((bs, L, K), "cpu")  # per-block redraw (lo=0 -> whole sequence)
    for s, t in uniform_schedule(steps):
        z = m.xst(z, torch.full((bs,), s), torch.full((bs,), t))
    assert torch.equal(blockwise, z.argmax(-1))


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


def test_renoise_single_step_carries_no_prefix_signal():
    """At s=0 a re-noised prefix is pure noise, so 1-step blocks ignore the prefix."""
    L, bs = 16, 4
    m = FakeCFM(L)
    gold_a = torch.zeros(bs, L, dtype=torch.long)
    gold_b = torch.full((bs, L), K - 1, dtype=torch.long)
    torch.manual_seed(0)
    out_a = blockwise_sample(m, 4, 1, batch_size=bs, length=L,
                             prefix_mode="renoise", gold_prefix=gold_a)
    torch.manual_seed(0)
    out_b = blockwise_sample(m, 4, 1, batch_size=bs, length=L,
                             prefix_mode="renoise", gold_prefix=gold_b)
    assert torch.equal(out_a, out_b)


def test_renoise_multi_step_conditions():
    """With >1 steps the second jump sees a partially-clean prefix — signal flows."""
    L, bs = 16, 4
    m = FakeCFM(L)
    gold_a = torch.zeros(bs, L, dtype=torch.long)
    gold_b = torch.full((bs, L), K - 1, dtype=torch.long)
    torch.manual_seed(0)
    out_a = blockwise_sample(m, 4, 2, batch_size=bs, length=L,
                             prefix_mode="renoise", gold_prefix=gold_a)
    torch.manual_seed(0)
    out_b = blockwise_sample(m, 4, 2, batch_size=bs, length=L,
                             prefix_mode="renoise", gold_prefix=gold_b)
    assert not torch.equal(out_a[:, 4:], out_b[:, 4:])


def test_sample_discretize_runs():
    m = FakeCFM(16)
    toks = blockwise_sample(m, 4, 2, batch_size=3, length=16, discretize="sample")
    assert toks.shape == (3, 16)
    assert 0 <= int(toks.min()) and int(toks.max()) < K


class FakeMaskedNet:
    """Minimal ``module.net`` for masked sampling tests (no flash-attn)."""

    def __init__(self, length: int, k: int = K):
        self.length = length
        g = torch.Generator().manual_seed(11)
        self.W = torch.randn(k, k, generator=g)

    def __call__(self, x, s, t, attn_mask=None):
        # x: (B, 2L, K) — only use noisy half pooled with masked clean prefix
        B, N, K_ = x.shape
        L = N // 2
        clean, noisy = x[:, :L], x[:, L:]
        if attn_mask is not None:
            # prefix influence: mean of clean positions the mask would allow is approximated
            ctx = clean.mean(dim=1, keepdim=True)
        else:
            ctx = x.mean(dim=1, keepdim=True)
        logits = noisy @ self.W + ctx.expand_as(noisy) + s.view(-1, 1, 1)
        return torch.cat([torch.zeros(B, L, K_, dtype=x.dtype), logits], dim=1)


class FakeMaskedCFM(FakeCFM):
    def __init__(self, length: int, k: int = K):
        super().__init__(length, k)
        self.net = FakeMaskedNet(length, k)


def test_masked_sample_shapes():
    m = FakeMaskedCFM(32)
    from block.sampling import blockwise_sample
    toks = blockwise_sample(m, block_size=8, steps_per_block=2, batch_size=4, length=32)
    assert toks.shape == (4, 32) and toks.dtype == torch.long
