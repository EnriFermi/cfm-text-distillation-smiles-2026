"""Current-block-only inference for checkpoint-compatible :class:`BlockDIT`.

This module is deliberately separate from the training implementation.  It adds no
parameters or buffers to ``BlockDIT`` and does not change checkpoint loading.  The
fast path evaluates the same block-causal graph incrementally:

* finalized clean blocks are encoded once and their per-layer K/V are cached;
* a noisy block evaluates only its ``block_size`` queries, attending to the cached
  strictly-earlier clean prefix and its own noisy K/V;
* after discretization, a non-terminal clean block is encoded once for successors.

The algebra is identical to the doubled-stream mask in ``block.mask``.  Floating
point results need not be bitwise identical because the ordinary path uses compiled
FlexAttention while this small dense path uses PyTorch SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from block.sampling import _discretize, uniform_schedule
from semicat.net.duo import modulate_fused


@dataclass
class CleanPrefixKVCache:
    """Preallocated clean-prefix K/V cache for one sampling batch."""

    keys: list[Tensor | None]
    values: list[Tensor | None]
    seq_len: int
    max_length: int
    batch_size: int

    @classmethod
    def empty(cls, net, batch_size: int) -> "CleanPrefixKVCache":
        n_layers = len(net.blocks)
        return cls(
            keys=[None] * n_layers,
            values=[None] * n_layers,
            seq_len=0,
            max_length=int(net.length),
            batch_size=int(batch_size),
        )

    def materialize(self, net, *, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate fixed-length zero-filled storage (needed by the compiled path)."""
        head_dim = net.hidden_size // net.n_heads
        for layer_index in range(len(net.blocks)):
            if self.keys[layer_index] is None:
                self.keys[layer_index] = torch.zeros(
                    self.batch_size, self.max_length, net.n_heads, head_dim,
                    device=device, dtype=dtype,
                )
                self.values[layer_index] = torch.zeros_like(self.keys[layer_index])


def _rotary_at(net, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    cos, sin = net.rotary_emb()
    return (
        cos[:, positions].to(dtype=dtype),
        sin[:, positions].to(dtype=dtype),
    )


def _block_qkv(net, layer, h: Tensor, rotary_cos_sin) -> tuple[Tensor, Tensor, Tensor]:
    batch, tokens, _ = h.shape
    qkv = layer.attn_qkv(h).reshape(
        batch,
        tokens,
        3,
        net.n_heads,
        net.hidden_size // net.n_heads,
    )
    return net._apply_rotary(qkv, rotary_cos_sin)


def _dense_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Unmasked attention for one block over ``[clean_prefix; current_block]``."""
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).flatten(2)


class FixedShapeFlowStep(nn.Module):
    """Compile-friendly flow step with a fixed ``L + block_size`` KV shape.

    Unfilled clean-cache slots are masked.  Keeping shapes invariant across block
    indices allows one compiled graph to serve all 16 TinyStories blocks.  Softmax
    and the Euler update live inside the compiled region as well: leaving them in
    Python costs several kernel launches for every one of ``n_blocks * NFE`` calls.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(
        self,
        noisy_block: Tensor,
        s: Tensor,
        t: Tensor,
        positions: Tensor,
        prefix_length: Tensor,
        step_scale: Tensor,
        clean_keys: tuple[Tensor, ...],
        clean_values: tuple[Tensor, ...],
    ) -> Tensor:
        cond = self.net._condition(s, t)
        h = self.net._embed(noisy_block, s, cond)
        rotary_cos_sin = _rotary_at(self.net, positions, h.dtype)
        prefix_allowed = torch.arange(
            self.net.length, device=noisy_block.device,
        ) < prefix_length
        allowed = torch.cat((
            prefix_allowed,
            torch.ones(noisy_block.shape[1], dtype=torch.bool, device=noisy_block.device),
        ))[None, None, None, :]

        for layer_index, layer in enumerate(self.net.blocks):
            modulation = self.net._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = _block_qkv(
                self.net, layer, h_norm, rotary_cos_sin,
            )
            key = torch.cat((clean_keys[layer_index], key), dim=1)
            value = torch.cat((clean_values[layer_index], value), dim=1)
            attention_output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2)
            h = self.net._finish_layer(
                layer, residual, attention_output, modulation,
            )
        logits = self.net._final_layer(h, cond)
        probabilities = logits.softmax(dim=-1)
        return noisy_block + step_scale * (probabilities - noisy_block)


class FixedShapeCleanBlock(nn.Module):
    """Compile-friendly in-place encoder for one finalized clean block."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(
        self,
        clean_block: Tensor,
        positions: Tensor,
        new_prefix_length: Tensor,
        clean_keys: tuple[Tensor, ...],
        clean_values: tuple[Tensor, ...],
    ) -> Tensor:
        batch, tokens, _ = clean_block.shape
        ones = torch.ones(
            batch, tokens, device=clean_block.device, dtype=clean_block.dtype,
        )
        cond = self.net._condition(ones, ones)
        h = self.net._embed(clean_block, ones, cond)
        rotary_cos_sin = _rotary_at(self.net, positions, h.dtype)
        allowed = (
            torch.arange(self.net.length, device=clean_block.device)
            < new_prefix_length
        )[None, None, None, :]

        for layer_index, layer in enumerate(self.net.blocks):
            modulation = self.net._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = _block_qkv(
                self.net, layer, h_norm, rotary_cos_sin,
            )
            clean_keys[layer_index].index_copy_(1, positions, key)
            clean_values[layer_index].index_copy_(1, positions, value)
            attention_output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                clean_keys[layer_index].transpose(1, 2),
                clean_values[layer_index].transpose(1, 2),
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2)
            h = self.net._finish_layer(
                layer, residual, attention_output, modulation,
            )
        # Returning the hidden block keeps the compiled side effects live.
        return h


_COMPILED_NOISY_BLOCKS: dict[int, nn.Module] = {}
_COMPILED_CLEAN_BLOCKS: dict[int, nn.Module] = {}


class DynamicPrefixFlowStep(nn.Module):
    """Flow step over exactly ``prefix_length + block_size`` keys.

    Unlike :class:`FixedShapeFlowStep`, this path has no padded cache and no mask.
    Inductor specializes the graph for each of the 16 prefix lengths on first use;
    that costs more warmup time but is the fastest steady-state batch=1 path.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(
        self,
        noisy_block: Tensor,
        s: Tensor,
        t: Tensor,
        positions: Tensor,
        step_scale: Tensor,
        clean_keys: tuple[Tensor, ...],
        clean_values: tuple[Tensor, ...],
    ) -> Tensor:
        cond = self.net._condition(s, t)
        h = self.net._embed(noisy_block, s, cond)
        rotary_cos_sin = _rotary_at(self.net, positions, h.dtype)
        for layer_index, layer in enumerate(self.net.blocks):
            modulation = self.net._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = _block_qkv(
                self.net, layer, h_norm, rotary_cos_sin,
            )
            key = torch.cat((clean_keys[layer_index], key), dim=1)
            value = torch.cat((clean_values[layer_index], value), dim=1)
            attention_output = _dense_attention(query, key, value)
            h = self.net._finish_layer(
                layer, residual, attention_output, modulation,
            )
        probabilities = self.net._final_layer(h, cond).softmax(dim=-1)
        return noisy_block + step_scale * (probabilities - noisy_block)


class DynamicPrefixCleanBlock(nn.Module):
    """Encode one finalized clean block and return only its new per-layer K/V."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(
        self,
        clean_block: Tensor,
        positions: Tensor,
        clean_keys: tuple[Tensor, ...],
        clean_values: tuple[Tensor, ...],
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        batch, tokens, _ = clean_block.shape
        ones = torch.ones(
            batch, tokens, device=clean_block.device, dtype=clean_block.dtype,
        )
        cond = self.net._condition(ones, ones)
        h = self.net._embed(clean_block, ones, cond)
        rotary_cos_sin = _rotary_at(self.net, positions, h.dtype)
        new_keys: list[Tensor] = []
        new_values: list[Tensor] = []
        for layer_index, layer in enumerate(self.net.blocks):
            modulation = self.net._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = _block_qkv(
                self.net, layer, h_norm, rotary_cos_sin,
            )
            attention_output = _dense_attention(
                query,
                torch.cat((clean_keys[layer_index], key), dim=1),
                torch.cat((clean_values[layer_index], value), dim=1),
            )
            h = self.net._finish_layer(
                layer, residual, attention_output, modulation,
            )
            new_keys.append(key)
            new_values.append(value)
        return tuple(new_keys), tuple(new_values)


_DYNAMIC_FLOW_STEPS: dict[int, nn.Module] = {}
_DYNAMIC_CLEAN_BLOCKS: dict[int, nn.Module] = {}


def compiled_dynamic_flow_step(net) -> nn.Module:
    key = id(net)
    scorer = _DYNAMIC_FLOW_STEPS.get(key)
    if scorer is None:
        scorer = torch.compile(
            DynamicPrefixFlowStep(net), fullgraph=True, mode="reduce-overhead",
        )
        _DYNAMIC_FLOW_STEPS[key] = scorer
    return scorer


def compiled_dynamic_clean_block(net) -> nn.Module:
    key = id(net)
    scorer = _DYNAMIC_CLEAN_BLOCKS.get(key)
    if scorer is None:
        scorer = torch.compile(
            DynamicPrefixCleanBlock(net), fullgraph=True, mode="reduce-overhead",
        )
        _DYNAMIC_CLEAN_BLOCKS[key] = scorer
    return scorer


def compiled_flow_step(net) -> nn.Module:
    """Return a process-local compiled flow step, cached by network identity."""
    key = id(net)
    scorer = _COMPILED_NOISY_BLOCKS.get(key)
    if scorer is None:
        scorer = torch.compile(
            FixedShapeFlowStep(net),
            fullgraph=True,
            mode="reduce-overhead",
        )
        _COMPILED_NOISY_BLOCKS[key] = scorer
    return scorer


def compiled_clean_block(net) -> nn.Module:
    """Return a process-local compiled clean encoder, cached by network identity."""
    key = id(net)
    scorer = _COMPILED_CLEAN_BLOCKS.get(key)
    if scorer is None:
        scorer = torch.compile(
            FixedShapeCleanBlock(net),
            fullgraph=True,
            mode="reduce-overhead",
        )
        _COMPILED_CLEAN_BLOCKS[key] = scorer
    return scorer


def _ensure_layer_storage(
    cache: CleanPrefixKVCache,
    layer_index: int,
    key: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    key_store = cache.keys[layer_index]
    value_store = cache.values[layer_index]
    if key_store is None or value_store is None:
        key_store = torch.empty(
            key.shape[0], cache.max_length, key.shape[2], key.shape[3],
            device=key.device, dtype=key.dtype,
        )
        value_store = torch.empty(
            value.shape[0], cache.max_length, value.shape[2], value.shape[3],
            device=value.device, dtype=value.dtype,
        )
        cache.keys[layer_index] = key_store
        cache.values[layer_index] = value_store
    return key_store, value_store


@torch.inference_mode()
def encode_clean_block(
    net,
    clean_block: Tensor,
    positions: Tensor,
    cache: CleanPrefixKVCache,
) -> None:
    """Append one finalized clean block to the per-layer prefix cache.

    ``positions`` must be the next contiguous absolute positions.  At every layer,
    queries from this block attend to all previous clean blocks plus the entire
    current clean block, matching ``clean -> clean: block(k) <= block(q)``.
    """
    batch, tokens, vocab = clean_block.shape
    if batch != cache.batch_size or vocab != net.vocab_size:
        raise ValueError("clean block and cache are incompatible")
    expected = torch.arange(
        cache.seq_len, cache.seq_len + tokens, device=positions.device,
    )
    if positions.shape != expected.shape or not torch.equal(positions, expected):
        raise ValueError("clean positions must be the next contiguous cache positions")
    if cache.seq_len + tokens > cache.max_length:
        raise ValueError("clean cache capacity exceeded")

    ones = torch.ones(batch, tokens, device=clean_block.device, dtype=clean_block.dtype)
    cond = net._condition(ones, ones)
    h = net._embed(clean_block, ones, cond)
    rotary_cos_sin = _rotary_at(net, positions, h.dtype)
    old_len = cache.seq_len
    new_len = old_len + tokens

    for layer_index, layer in enumerate(net.blocks):
        modulation = net._layer_modulation(layer, cond)
        shift_msa, scale_msa, *_ = modulation
        residual = h
        h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
        query, key, value = _block_qkv(net, layer, h_norm, rotary_cos_sin)
        key_store, value_store = _ensure_layer_storage(cache, layer_index, key, value)
        key_store[:, old_len:new_len].copy_(key)
        value_store[:, old_len:new_len].copy_(value)
        attention_output = _dense_attention(
            query, key_store[:, :new_len], value_store[:, :new_len],
        )
        h = net._finish_layer(layer, residual, attention_output, modulation)

    cache.seq_len = new_len


@torch.inference_mode()
def forward_noisy_block(
    net,
    noisy_block: Tensor,
    s: Tensor,
    t: Tensor,
    positions: Tensor,
    cache: CleanPrefixKVCache,
) -> Tensor:
    """Score one noisy block against a cached strictly-earlier clean prefix."""
    batch, tokens, vocab = noisy_block.shape
    if batch != cache.batch_size or vocab != net.vocab_size:
        raise ValueError("noisy block and cache are incompatible")
    if s.shape != (batch, tokens) or t.shape != s.shape:
        raise ValueError("noisy times must have shape (batch, block_tokens)")
    expected = torch.arange(cache.seq_len, cache.seq_len + tokens, device=positions.device)
    if positions.shape != expected.shape or not torch.equal(positions, expected):
        raise ValueError("noisy block must immediately follow the cached clean prefix")

    cond = net._condition(s, t)
    h = net._embed(noisy_block, s, cond)
    rotary_cos_sin = _rotary_at(net, positions, h.dtype)

    for layer_index, layer in enumerate(net.blocks):
        modulation = net._layer_modulation(layer, cond)
        shift_msa, scale_msa, *_ = modulation
        residual = h
        h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
        query, key, value = _block_qkv(net, layer, h_norm, rotary_cos_sin)
        if cache.seq_len:
            clean_key = cache.keys[layer_index]
            clean_value = cache.values[layer_index]
            if clean_key is None or clean_value is None:
                raise RuntimeError("partially initialized clean cache")
            key = torch.cat((clean_key[:, :cache.seq_len], key), dim=1)
            value = torch.cat((clean_value[:, :cache.seq_len], value), dim=1)
        attention_output = _dense_attention(query, key, value)
        h = net._finish_layer(layer, residual, attention_output, modulation)

    return net._final_layer(h, cond)


@torch.inference_mode()
def block_causal_sample_cached(
    module,
    block_size: int,
    steps_per_block: int,
    *,
    batch_size: int,
    length: int | None = None,
    schedule: list[tuple[float, float]] | None = None,
    discretize: Literal["argmax", "sample"] = "argmax",
    compile_noisy: bool = False,
    compile_strategy: Literal["dynamic", "fixed"] = "dynamic",
) -> Tensor:
    """Sample M3 using current-block-only execution and a clean-prefix KV cache."""
    length = int(length or module.in_shape[0])
    vocab = int(module.in_shape[-1])
    if length != module.net.length:
        raise ValueError("cached BlockDIT inference currently requires its training length")
    if length % block_size:
        raise ValueError(f"length {length} not divisible by block_size {block_size}")
    if block_size != module.net.block_size:
        raise ValueError("sampler block_size must match BlockDIT.block_size")

    device = module.device
    sched = [
        (float(s), float(t))
        for s, t in (schedule or uniform_schedule(steps_per_block))
    ]
    tokens = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    if compile_strategy not in {"dynamic", "fixed"}:
        raise ValueError(f"unknown compile_strategy={compile_strategy!r}")
    dynamic = compile_noisy and compile_strategy == "dynamic"
    cache = CleanPrefixKVCache.empty(module.net, batch_size)
    if dynamic:
        scorer = compiled_dynamic_flow_step(module.net)
        clean_scorer = compiled_dynamic_clean_block(module.net)
        head_dim = module.net.hidden_size // module.net.n_heads
        empty_shape = (batch_size, 0, module.net.n_heads, head_dim)
        cache.keys = [
            torch.empty(empty_shape, device=device, dtype=next(module.net.parameters()).dtype)
            for _ in module.net.blocks
        ]
        cache.values = [torch.empty_like(item) for item in cache.keys]
    else:
        scorer = compiled_flow_step(module.net) if compile_noisy else None
        clean_scorer = compiled_clean_block(module.net) if compile_noisy else None
    if scorer is not None and not dynamic:
        cache_dtype = next(module.net.parameters()).dtype
        if torch.is_autocast_enabled(device.type):
            cache_dtype = torch.get_autocast_dtype(device.type)
        cache.materialize(
            module.net,
            device=device,
            dtype=cache_dtype,
        )
    n_blocks = length // block_size

    for block_index in range(n_blocks):
        lo = block_index * block_size
        hi = lo + block_size
        positions = torch.arange(lo, hi, device=device)
        z_block = module.prior((batch_size, block_size, vocab), device=device)
        for s_value, t_value in sched:
            s = torch.full(
                (batch_size, block_size), s_value, device=device, dtype=z_block.dtype,
            )
            t = torch.full(
                (batch_size, block_size), t_value, device=device, dtype=z_block.dtype,
            )
            if scorer is None:
                logits = forward_noisy_block(
                    module.net, z_block, s, t, positions, cache,
                )
            elif dynamic:
                if any(item is None for item in cache.keys + cache.values):
                    raise RuntimeError("dynamic cache must be materialized")
                clean_keys = tuple(cache.keys)
                clean_values = tuple(cache.values)
                step_scale = torch.scalar_tensor(
                    (t_value - s_value) / (1.0 - s_value + 1e-8),
                    device=device,
                    dtype=z_block.dtype,
                )
                # reduce-overhead owns static CUDA-graph output buffers.  Clone the
                # endpoint before the next replay can overwrite it.
                z_block = scorer(
                    z_block, s, t, positions, step_scale,
                    clean_keys, clean_values,
                ).clone()
            else:
                if any(item is None for item in cache.keys + cache.values):
                    raise RuntimeError("compiled scorer requires materialized cache")
                step_scale = torch.scalar_tensor(
                    (t_value - s_value) / (1.0 - s_value + 1e-8),
                    device=device,
                    dtype=z_block.dtype,
                )
                z_block = scorer(
                    z_block,
                    s,
                    t,
                    positions,
                    torch.scalar_tensor(cache.seq_len, device=device, dtype=torch.long),
                    step_scale,
                    tuple(cache.keys),
                    tuple(cache.values),
                )
            if scorer is None:
                q_block = logits.softmax(dim=-1)
                z_block = z_block + (
                    (t_value - s_value) / (1.0 - s_value + 1e-8)
                ) * (q_block - z_block)
        block = _discretize(z_block, discretize)
        tokens[:, lo:hi] = block
        if block_index + 1 < n_blocks:
            clean_block = F.one_hot(block, vocab).to(z_block.dtype)
            if clean_scorer is None:
                encode_clean_block(module.net, clean_block, positions, cache)
            elif dynamic:
                if any(item is None for item in cache.keys + cache.values):
                    raise RuntimeError("dynamic cache must be materialized")
                clean_keys = tuple(cache.keys)
                clean_values = tuple(cache.values)
                new_keys, new_values = clean_scorer(
                    clean_block, positions, clean_keys, clean_values,
                )
                # Compiled reduce-overhead outputs alias static graph buffers.
                for layer_index in range(len(cache.keys)):
                    key = new_keys[layer_index].clone()
                    value = new_values[layer_index].clone()
                    cache.keys[layer_index] = torch.cat(
                        (clean_keys[layer_index], key), dim=1,
                    )
                    cache.values[layer_index] = torch.cat(
                        (clean_values[layer_index], value), dim=1,
                    )
                cache.seq_len = hi
            else:
                if any(item is None for item in cache.keys + cache.values):
                    raise RuntimeError("compiled encoder requires materialized cache")
                clean_scorer(
                    clean_block,
                    positions,
                    torch.scalar_tensor(hi, device=device, dtype=torch.long),
                    tuple(cache.keys),
                    tuple(cache.values),
                )
                cache.seq_len = hi

    return tokens
