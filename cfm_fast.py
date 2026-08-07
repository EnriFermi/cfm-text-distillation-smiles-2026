"""An optimized-but-equivalent CFM block-causal sampler.

`block/sampling.py::block_causal_sample` is written for clarity, not speed, and
leaves three exact wins on the table. All of them exploit the fact that the two
vocab-sized projections dominate the cost: at vocab 50257 vs hidden 384, the
input `proj` and the output head are ~40 GFLOP of the ~50 GFLOP per jump, while
the six transformer layers are only ~11.

1. **Clean stream as an embedding lookup.** The reference materializes
   `F.one_hot(tokens, 50257)` and matmuls it through `proj`. For a one-hot input
   with a scalar time scale that product *is* a row lookup:
   `proj(one_hot(i) * c) == c * W[:, i] + bias`. Exact, and it also stops the
   `(batch, 256, 50257)` tensor from ever existing.

2. **Noisy stream only on the active block.** Every noisy position outside
   `[lo, hi)` is exactly zero, and `proj(0) == bias`, so those rows are a
   broadcast of the bias rather than a matmul.

3. **Vocab head only on the active block.** `_final_layer` is token-wise
   (LayerNorm over the hidden dim, per-token adaLN, per-token linear), so
   slicing before it changes nothing about the 16 rows we keep — the reference
   computes all 512 and throws 496 away.

`assert_matches_reference` checks 1-3 against the untouched sampler; they are
algebraic identities, so fp32 agreement is to within accumulation-order noise.

bf16 autocast and `torch.compile` are the two non-exact knobs and stay opt-in.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor

from semicat.net.duo import RMSEmbeddingLayer, modulate_fused


def prepare(net) -> None:
    """Cache the transposed vocab projection used by the lookup path."""
    if not isinstance(net.vocab_embed, RMSEmbeddingLayer):
        raise TypeError("fast sampler expects the RMS embedding used by this checkpoint")
    if not hasattr(net, "_fast_proj_wt"):
        net._fast_proj_wt = net.vocab_embed.proj.weight.t().contiguous()


def _embed(net, tokens: Tensor, z_blk: Tensor, lo: int, hi: int,
           s_full: Tensor, cond: Tensor) -> Tensor:
    """Token-wise embedding of `[clean ; noisy]` without either dense matmul."""
    embed = net.vocab_embed
    batch, length = tokens.shape

    # clean half: one-hot rows at s = 1 -> expected_norm2 = 1
    clean_scale = (1.0 + embed.eps) ** -0.5
    h_clean = F.embedding(tokens, net._fast_proj_wt) * clean_scale + embed.proj.bias

    # noisy half: zero outside the active block, and proj(0) is just the bias
    s_blk = s_full[:, length + lo: length + hi]
    expected = (s_blk.square()
                + (1.0 - s_blk).square() * (embed.vocab_dim * embed.sigma0 ** 2))
    blk_scale = torch.rsqrt(expected + embed.eps)[..., None]
    h_noisy = embed.proj.bias.to(h_clean.dtype).expand(batch, length, -1).contiguous()
    h_noisy[:, lo:hi] = embed.proj(z_blk * blk_scale)

    h = torch.cat((h_clean, h_noisy), dim=1)
    h = h + embed.residual_scale * embed.mlp(h)
    h = (1.0 + embed.film_gamma(cond)) * h + embed.film_beta(cond)
    return embed.final_norm(h)


def make_body(net, compile_body: bool = False):
    """The six transformer layers as one callable `(h, cond) -> h`.

    Only the body is worth compiling: it has fixed `(batch, 2L, hidden)` shapes,
    whereas the surrounding code is parameterized by `lo/hi/s/t`, which change
    every step and would re-trigger a guard on each of the `16 * steps` calls.
    """
    def body(h: Tensor, cond: Tensor) -> Tensor:
        rotary_cos_sin = net.rotary_emb()
        for layer in net.blocks:
            modulation = net._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = net._doubled_qkv(layer, h_norm, rotary_cos_sin)
            attention = net._normal_attention(query, key, value, clean_only=False)
            h = net._finish_layer(layer, residual, attention, modulation)
        return h

    return torch.compile(body) if compile_body else body


def active_forward(net, tokens: Tensor, z_blk: Tensor, lo: int, hi: int,
                   s: float, t: float, body=None) -> Tensor:
    """Logits for the active noisy block only — `(batch, block_size, vocab)`."""
    batch, length = tokens.shape
    device = tokens.device
    ones = torch.ones(batch, length, device=device)
    s_tok = torch.zeros(batch, length, device=device)
    t_tok = torch.zeros(batch, length, device=device)
    s_tok[:, lo:hi], t_tok[:, lo:hi] = s, t
    s_full = torch.cat((ones, s_tok), dim=1)
    t_full = torch.cat((ones, t_tok), dim=1)

    cond = net._condition(s_full, t_full)
    h = _embed(net, tokens, z_blk, lo, hi, s_full, cond)
    h = (body or make_body(net))(h, cond)

    active = slice(length + lo, length + hi)
    return net._final_layer(h[:, active], cond[:, active])


@torch.inference_mode()
def fast_block_causal_sample(module, block_size: int, steps_per_block: int, *,
                             batch_size: int, length: int | None = None,
                             discretize: str = "argmax",
                             amp_dtype: torch.dtype | None = None,
                             body=None) -> Tensor:
    """Drop-in replacement for `block_causal_sample` on a `BlockSemicatModule`."""
    net = module.net
    prepare(net)
    body = body or make_body(net)
    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    grid = torch.linspace(0.0, 1.0, steps_per_block + 1).tolist()
    schedule = list(zip(grid[:-1], grid[1:]))

    autocast = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype
                else contextlib.nullcontext())
    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    for index in range(L // block_size):
        lo, hi = index * block_size, (index + 1) * block_size
        z_blk = module.prior((batch_size, block_size, K), device=device)
        for s, t in schedule:
            with autocast:
                logits = active_forward(net, tokens, z_blk, lo, hi, s, t, body=body)
            q_blk = logits.float().softmax(dim=-1)
            z_blk = z_blk + ((t - s) / (1.0 - s + 1e-8)) * (q_blk - z_blk)
        tokens[:, lo:hi] = (z_blk.argmax(-1) if discretize == "argmax" else
                            torch.multinomial(z_blk.clamp_min(0).flatten(0, 1), 1)
                            .squeeze(-1).view(batch_size, block_size))
    return tokens


# --------------------------------------------------------------- clean KV cache
# `block/mask.py::block_causal_mask` grants exactly three attention edges:
#   clean -> clean   block_k <= block_q   (block-causal: earlier blocks are final)
#   noisy -> clean   block_k <  block_q   (strictly previous)
#   noisy -> noisy   block_k == block_q   (its own block, unmasked within it)
# Two consequences make a KV cache exact here, the same way it is for BD3-LM.
# First, a clean position's representation never depends on a later block, so the
# clean stream can be extended one block at a time and never rewritten. Second,
# the clean stream carries s = t = 1 everywhere, so it does not depend on the
# flow-map times at all — the reference sampler rebuilds it identically on all
# `steps_per_block` jumps of a block.
#
# So: keep per-layer clean K/V for the finalized blocks, and run each jump as a
# `block_size`-token noisy forward attending to `[cached clean ; own block]` with
# no mask. That is 16 query positions over <=256 keys instead of 512 over 512.


def _rotary_at(net, lo: int, hi: int):
    """RoPE cache restricted to absolute positions `[lo, hi)`."""
    cos, sin = net.rotary_emb()
    return cos[:, lo:hi], sin[:, lo:hi]


def _qkv_at(net, layer, h: Tensor, rotary):
    """`_stream_qkv` without its full-length assertion."""
    batch, tokens, _ = h.shape
    qkv = layer.attn_qkv(h).reshape(batch, tokens, 3, net.n_heads,
                                    net.hidden_size // net.n_heads)
    return net._apply_rotary(qkv, rotary)


def _embed_block(net, x: Tensor, s: Tensor, cond: Tensor, *, lookup: bool) -> Tensor:
    """RMS embedding for one block; `lookup` takes the one-hot shortcut."""
    embed = net.vocab_embed
    if lookup:
        h = F.embedding(x, net._fast_proj_wt) * (1.0 + embed.eps) ** -0.5 + embed.proj.bias
    else:
        expected = (s.square()
                    + (1.0 - s).square() * (embed.vocab_dim * embed.sigma0 ** 2))
        h = embed.proj(x * torch.rsqrt(expected + embed.eps)[..., None])
    h = h + embed.residual_scale * embed.mlp(h)
    h = (1.0 + embed.film_gamma(cond)) * h + embed.film_beta(cond)
    return embed.final_norm(h)


def new_cache(net) -> dict:
    return {"keys": [None] * len(net.blocks), "values": [None] * len(net.blocks)}


def extend_cache(net, cache: dict, tokens: Tensor, lo: int, hi: int) -> None:
    """Append the just-finalized clean block `[lo, hi)` to the cache."""
    batch = tokens.shape[0]
    ones = torch.ones(batch, hi - lo, device=tokens.device)
    cond = net._condition(ones, ones)
    h = _embed_block(net, tokens[:, lo:hi], ones, cond, lookup=True)
    rotary = _rotary_at(net, lo, hi)

    for index, layer in enumerate(net.blocks):
        modulation = net._layer_modulation(layer, cond)
        shift_msa, scale_msa, *_ = modulation
        residual = h
        h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
        query, key, value = _qkv_at(net, layer, h_norm, rotary)
        # clean -> clean is block_k <= block_q: the cache (strictly earlier
        # blocks) plus this block itself, with no mask inside the block.
        if cache["keys"][index] is not None:
            key = torch.cat((cache["keys"][index], key), dim=1)
            value = torch.cat((cache["values"][index], value), dim=1)
        attention = net._sdpa(query, key, value, mask=None, math_only=False)
        h = net._finish_layer(layer, residual, attention, modulation)
        cache["keys"][index], cache["values"][index] = key, value


def cached_forward(net, cache: dict, z_blk: Tensor, lo: int, hi: int,
                   s: float, t: float) -> Tensor:
    """One flow-map jump on the active block against the cached clean prefix."""
    batch, block, _ = z_blk.shape
    s_tok = torch.full((batch, block), s, device=z_blk.device)
    t_tok = torch.full((batch, block), t, device=z_blk.device)
    cond = net._condition(s_tok, t_tok)
    h = _embed_block(net, z_blk, s_tok, cond, lookup=False)
    rotary = _rotary_at(net, lo, hi)

    for index, layer in enumerate(net.blocks):
        modulation = net._layer_modulation(layer, cond)
        shift_msa, scale_msa, *_ = modulation
        residual = h
        h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
        query, key, value = _qkv_at(net, layer, h_norm, rotary)
        # noisy -> clean is strictly previous, and the cache holds exactly the
        # finalized blocks, so it needs no slicing; noisy -> noisy is this block.
        if cache["keys"][index] is not None:
            key = torch.cat((cache["keys"][index], key), dim=1)
            value = torch.cat((cache["values"][index], value), dim=1)
        attention = net._sdpa(query, key, value, mask=None, math_only=False)
        h = net._finish_layer(layer, residual, attention, modulation)
    return net._final_layer(h, cond)


@torch.inference_mode()
def cached_block_causal_sample(module, block_size: int, steps_per_block: int, *,
                               batch_size: int, length: int | None = None,
                               discretize: str = "argmax",
                               amp_dtype: torch.dtype | None = None) -> Tensor:
    """`fast_block_causal_sample` plus the clean-prefix KV cache."""
    net = module.net
    prepare(net)
    L = length or module.in_shape[0]
    K = module.in_shape[-1]
    if L % block_size:
        raise ValueError(f"length {L} not divisible by block_size {block_size}")
    device = module.device
    grid = torch.linspace(0.0, 1.0, steps_per_block + 1).tolist()
    schedule = list(zip(grid[:-1], grid[1:]))
    autocast = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype
                else contextlib.nullcontext())

    tokens = torch.zeros(batch_size, L, dtype=torch.long, device=device)
    cache = new_cache(net)
    for index in range(L // block_size):
        lo, hi = index * block_size, (index + 1) * block_size
        z_blk = module.prior((batch_size, block_size, K), device=device)
        for s, t in schedule:
            with autocast:
                logits = cached_forward(net, cache, z_blk, lo, hi, s, t)
            q_blk = logits.float().softmax(dim=-1)
            z_blk = z_blk + ((t - s) / (1.0 - s + 1e-8)) * (q_blk - z_blk)
        tokens[:, lo:hi] = (z_blk.argmax(-1) if discretize == "argmax" else
                            torch.multinomial(z_blk.clamp_min(0).flatten(0, 1), 1)
                            .squeeze(-1).view(batch_size, block_size))
        with autocast:
            extend_cache(net, cache, tokens, lo, hi)
    return tokens


@torch.inference_mode()
def assert_matches_reference(module, block_size: int, steps: int, batch_size: int = 2,
                             cached: bool = False) -> dict:
    """Same seed, reference vs optimized, in fp32: tokens must be identical."""
    from block.sampling import block_causal_sample

    def seeded(fn):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        return fn()

    reference = seeded(lambda: block_causal_sample(
        module, block_size=block_size, steps_per_block=steps,
        batch_size=batch_size, discretize="argmax"))
    sampler = cached_block_causal_sample if cached else fast_block_causal_sample
    fast = seeded(lambda: sampler(module, block_size, steps,
                                  batch_size=batch_size, discretize="argmax"))
    agree = (reference == fast).float().mean().item()
    return {"token_agreement": agree, "identical": bool(agree == 1.0),
            "reference_head": reference[0, :16].tolist(),
            "fast_head": fast[0, :16].tolist()}
