"""Tests for checkpoint-compatible training-time BCFM."""

import pytest
import torch
import torch.nn.functional as F

from block.block_dit import BlockDIT
from block.block_semicat import BlockSemicatModule
from block.mask import block_causal_mask, block_causal_mask_mod
from block.sampling import block_causal_sample
from semicat.models.semicat import SemicatModule
from semicat.net.duo import DIT

L, B, K = 8, 2, 5          # length, block_size, vocab -> 4 blocks, doubled length 16
NB = L // B


def _blk(pos):  # block id within a stream
    return (pos % L) // B


# --------------------------------------------------------------------------- mask
def test_mask_shape():
    m = block_causal_mask(L, B)
    assert m.shape == (2 * L, 2 * L) and m.dtype == torch.bool


def test_flex_mask_mod_is_exactly_the_dense_mask():
    query = torch.arange(2 * L)[:, None]
    key = torch.arange(2 * L)[None, :]
    flex_form = block_causal_mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        query,
        key,
        length=L,
        block_size=B,
    )
    assert torch.equal(flex_form, block_causal_mask(L, B))


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
                    n_heads=2, dropout=0.0, length=L, block_size=B,
                    attention_backend="sdpa", jvp_attention_backend="math")


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
    q, dq = torch.func.jvp(lambda _t: net(x, s, _t, mask, jvp_attention=True).softmax(-1),
                           (t,), (torch.ones_like(t),))
    assert q.shape == dq.shape == (2, 2 * L, K)
    assert torch.isfinite(q).all() and torch.isfinite(dq).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton JVP requires CUDA")
def test_triton_block_jvp_backward_batch_greater_than_one():
    """Regression: block slices must be compact for the custom JVP backward."""
    net = BlockDIT(
        vocab_size=K,
        hidden_size=32,
        cond_dim=8,
        n_blocks=1,
        n_heads=2,
        dropout=0.0,
        length=L,
        block_size=B,
        attention_backend="sdpa",
        jvp_attention_backend="triton",
    ).cuda()
    clean = torch.randn(2, L, K, device="cuda").softmax(-1)
    noisy = torch.randn(2, L, K, device="cuda").softmax(-1)
    s = torch.rand(2, L, device="cuda")
    t = s + torch.rand_like(s) * (1.0 - s)
    cache = net.build_clean_kv_cache(clean)
    prediction, tangent = torch.func.jvp(
        lambda target: net.forward_noisy_jvp(noisy, s, target, cache).softmax(-1),
        (t,),
        (torch.ones_like(t),),
    )
    (prediction.square().mean() + tangent.square().mean()).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in net.parameters()
    )


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
def _tiny_module(sd_type="lag"):
    return BlockSemicatModule(
        net=_tiny_net(), optimizer=None, scheduler=None,
        in_shape=(L, K), prior_type="gaussian", sd_prop=0.5, block_size=B,
        sd_type=sd_type,
    )


def test_cfm_alias_selects_upstream_lagrangian_loss():
    assert _tiny_module("cfm").hparams.sd_type == "lag"


@pytest.mark.parametrize("sd_type", ["lag", "ecld"])
def test_training_step_finite(sd_type):
    m = _tiny_module(sd_type)
    x1 = torch.randint(0, K, (4, L))
    vf, sd = m.model_step(x1)
    assert torch.isfinite(vf) and vf > 0
    assert sd is not None and torch.isfinite(sd)
    (vf + sd).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in m.net.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="upstream JVP requires CUDA")
@pytest.mark.parametrize("sd_type", ["lag", "ecld"])
def test_block_size_length_preserves_upstream_cfm_losses(sd_type):
    """At B=L the block formulation reduces numerically to untouched CFM."""
    kwargs = dict(
        vocab_size=K,
        hidden_size=32,
        cond_dim=8,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
        length=L,
        embed_type="naive",
    )
    duo = DIT(**kwargs).cuda()
    block_net = BlockDIT(
        **kwargs,
        block_size=L,
        attention_backend="sdpa",
        jvp_attention_backend="triton",
    ).cuda()
    block_net.load_state_dict(duo.state_dict(), strict=True)
    upstream = SemicatModule(
        net=duo,
        optimizer=None,
        scheduler=None,
        in_shape=(L, K),
        prior_type="gaussian",
        sd_prop=0.5,
        sd_type=sd_type,
    )
    block = BlockSemicatModule(
        net=block_net,
        optimizer=None,
        scheduler=None,
        in_shape=(L, K),
        prior_type="gaussian",
        sd_prop=0.5,
        sd_type=sd_type,
        block_size=L,
    )
    batch = torch.randint(0, K, (4, L), device="cuda")
    torch.manual_seed(29)
    upstream_vf, upstream_sd = upstream.model_step(batch)
    torch.manual_seed(29)
    block_vf, block_sd = block.model_step(batch)
    assert upstream_sd is not None and block_sd is not None
    assert torch.allclose(block_vf, upstream_vf, atol=2e-5, rtol=2e-5)
    assert torch.allclose(block_sd, upstream_sd, atol=2e-5, rtol=2e-5)


def test_block_causal_sample():
    m = _tiny_module().eval()
    toks = block_causal_sample(m, B, steps_per_block=2, batch_size=3, length=L)
    assert toks.shape == (3, L) and toks.dtype == torch.long
    assert 0 <= int(toks.min()) and int(toks.max()) < K


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("discretize", ["argmax", "sample"])
@pytest.mark.parametrize("backend", ["cached", "fused_cached"])
def test_cached_block_causal_sample_matches_full_on_cpu(steps, discretize, backend):
    """The KV path preserves the doubled-stream sampler on a deterministic CPU net."""
    m = _tiny_module().eval()
    _dezero(m.net)
    torch.manual_seed(731)
    expected = block_causal_sample(
        m, B, steps_per_block=steps, batch_size=3, length=L,
        discretize=discretize, inference_backend="full",
    )
    torch.manual_seed(731)
    actual = block_causal_sample(
        m, B, steps_per_block=steps, batch_size=3, length=L,
        discretize=discretize, inference_backend=backend,
    )
    assert torch.equal(actual, expected)


def test_sample_flow_map_batch_onehot():
    m = _tiny_module().eval()
    out = m.sample_flow_map_batch(batch_size=2, sampling_steps=1)
    assert out.shape == (2, L, K)
    assert torch.equal(out.sum(-1), torch.ones(2, L))  # valid one-hots


def test_state_dict_namespace_is_exactly_duo_compatible():
    kwargs = dict(
        vocab_size=K,
        hidden_size=16,
        cond_dim=8,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
        length=L,
        embed_type="naive",
    )
    duo = DIT(**kwargs)
    block = BlockDIT(
        **kwargs,
        block_size=B,
        attention_backend="sdpa",
        jvp_attention_backend="math",
    )
    assert list(block.state_dict()) == list(duo.state_dict())
    block.load_state_dict(duo.state_dict(), strict=True)


def test_block_size_length_preserves_duo_noisy_logits():
    torch.manual_seed(11)
    kwargs = dict(
        vocab_size=K,
        hidden_size=16,
        cond_dim=8,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
        length=L,
        embed_type="naive",
    )
    duo = _dezero(DIT(**kwargs)).eval()
    block = BlockDIT(
        **kwargs,
        block_size=L,
        attention_backend="sdpa",
        jvp_attention_backend="math",
    ).eval()
    block.load_state_dict(duo.state_dict(), strict=True)

    noisy = torch.randn(2, L, K)
    clean = F.one_hot(torch.randint(0, K, (2, L)), K).float()
    s = torch.rand(2)
    t = s + torch.rand(2) * (1.0 - s)
    duo_logits = duo(noisy, s, t)
    doubled_s = torch.cat((torch.ones(2, L), s[:, None].expand(-1, L)), dim=1)
    doubled_t = torch.cat((torch.ones(2, L), t[:, None].expand(-1, L)), dim=1)
    block_logits = block(
        torch.cat((clean, noisy), dim=1),
        doubled_s,
        doubled_t,
    )[:, L:]
    assert torch.allclose(block_logits, duo_logits, atol=2e-5, rtol=2e-5)
