"""Checkpoint-compatible block-causal DiT for BCFM fine-tuning.

The trainable architecture and state-dict namespace are deliberately identical
to :class:`semicat.net.duo.DIT`. The only changes are the BCFM formulation:

* a doubled ``[clean ; noisy]`` sequence;
* a BD3-LM block-causal attention graph;
* one ``(s_b, t_b)`` pair per block instead of one pair per sample.

Ordinary VFM/teacher forwards use the compiled FlexAttention pattern from the
official BD3-LM implementation. The CFM ``lag`` and ECLD objectives additionally
need forward-mode AD through attention. PyTorch FlexAttention does not support
JVP, so that path precomputes clean-stream KV once and applies SemiCat's original
custom Triton JVP attention independently to each noisy block and its permitted
clean prefix. CPU/debug execution falls back to JVP-safe math SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from block.mask import (
    FLEX_ATTENTION_AVAILABLE,
    block_causal_mask,
    create_block_causal_flex_mask,
    create_clean_causal_flex_mask,
)
from semicat.jvp_utils.functional import safe_sdpa_jvp
from semicat.net.duo import (
    DIT as DuoDIT,
    EmbeddingLayer,
    RMSEmbeddingLayer,
    modulate_fused,
    split_and_apply_rotary_pos_emb,
)

if FLEX_ATTENTION_AVAILABLE:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    # This follows the official BD3-LM compiled FlexAttention wrapper
    # (kuleshov-group/bd3lms, Apache-2.0, commit 1c3e8f43...).
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def _compiled_flex_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: BlockMask,
    ) -> Tensor:
        return flex_attention(query, key, value, block_mask=mask)


AttentionBackend = Literal["flex", "sdpa"]
JVPAttentionBackend = Literal["auto", "triton", "math"]


@dataclass(frozen=True)
class CleanKVCache:
    """Per-layer clean-stream keys and values used by the sparse JVP path."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    batch_size: int
    length: int


class BlockDIT(DuoDIT):
    """A block-causal execution of the exact upstream ``duo.DIT`` parameters.

    No trainable module is added, removed, or renamed. Consequently a
    full-sequence CFM checkpoint can be strict-loaded after stripping the
    Lightning ``net.`` prefix.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        length: int,
        block_size: int,
        embed_type: Literal["naive", "rms"] = "naive",
        attention_backend: AttentionBackend = "flex",
        jvp_attention_backend: JVPAttentionBackend = "auto",
        flex_kernel_block_size: int = 64,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            cond_dim=cond_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            length=length,
            embed_type=embed_type,
        )
        if length % block_size:
            raise ValueError(f"length {length} is not divisible by block_size {block_size}")
        if attention_backend not in {"flex", "sdpa"}:
            raise ValueError(f"unknown attention_backend={attention_backend!r}")
        if jvp_attention_backend not in {"auto", "triton", "math"}:
            raise ValueError(f"unknown jvp_attention_backend={jvp_attention_backend!r}")
        if flex_kernel_block_size <= 0:
            raise ValueError("flex_kernel_block_size must be positive")

        self.length = int(length)
        self.block_size = int(block_size)
        self.hidden_size = int(hidden_size)
        self.cond_dim = int(cond_dim)
        self.n_heads = int(n_heads)
        self.attention_backend = attention_backend
        self.jvp_attention_backend = jvp_attention_backend
        self.flex_kernel_block_size = int(flex_kernel_block_size)

        # Runtime-only caches: no buffers/parameters are introduced, preserving
        # exact state_dict compatibility with DuoDIT.
        self._flex_masks: dict[tuple[str, int | None, str], object] = {}
        self._warned_flex_fallback = False

    # ------------------------------------------------------------------ embeddings
    def _condition(self, s: Tensor, t: Tensor) -> Tensor:
        if s.shape != t.shape or s.ndim != 2:
            raise ValueError(f"expected matching (batch, tokens) times, got {s.shape}, {t.shape}")
        batch, tokens = s.shape
        s_cond = F.silu(self.s_map(s.reshape(-1)))
        t_cond = F.silu(self.t_map((t - s).reshape(-1)))
        return (s_cond + t_cond).reshape(batch, tokens, self.cond_dim)

    def _embed(self, x: Tensor, s: Tensor, cond: Tensor) -> Tensor:
        if isinstance(self.vocab_embed, EmbeddingLayer):
            return self.vocab_embed(x, s, cond)
        if not isinstance(self.vocab_embed, RMSEmbeddingLayer):
            raise TypeError(f"unsupported upstream embedding {type(self.vocab_embed).__name__}")

        # Exact token-wise generalization of upstream RMSEmbeddingLayer.forward.
        expected_norm2 = (
            s.square()
            + (1.0 - s).square()
            * (self.vocab_embed.vocab_dim * self.vocab_embed.sigma0**2)
        )
        scale = torch.rsqrt(expected_norm2 + self.vocab_embed.eps)[..., None]
        h = self.vocab_embed.proj(x * scale)
        h = h + self.vocab_embed.residual_scale * self.vocab_embed.mlp(h)
        h = (
            (1.0 + self.vocab_embed.film_gamma(cond)) * h
            + self.vocab_embed.film_beta(cond)
        )
        return self.vocab_embed.final_norm(h)

    # ---------------------------------------------------------------------- masks
    @staticmethod
    def _device_key(device: torch.device) -> tuple[str, int | None]:
        return device.type, device.index

    def _flex_mask(self, device: torch.device, clean_only: bool) -> object:
        kind = "clean" if clean_only else "doubled"
        key = (*self._device_key(device), kind)
        if key not in self._flex_masks:
            factory = (
                create_clean_causal_flex_mask
                if clean_only
                else create_block_causal_flex_mask
            )
            self._flex_masks[key] = factory(
                self.length,
                self.block_size,
                device,
                kernel_block_size=self.flex_kernel_block_size,
            )
        return self._flex_masks[key]

    def _dense_clean_mask(self, device: torch.device) -> Tensor:
        blocks = torch.arange(self.length, device=device) // self.block_size
        return blocks[None, :] <= blocks[:, None]

    # --------------------------------------------------------------- transformer ops
    def _stream_qkv(self, layer: nn.Module, h: Tensor, rotary_cos_sin) -> tuple[Tensor, Tensor, Tensor]:
        batch, tokens, _ = h.shape
        if tokens != self.length:
            raise ValueError(f"single stream must have length {self.length}, got {tokens}")
        qkv = layer.attn_qkv(h).reshape(
            batch,
            tokens,
            3,
            self.n_heads,
            self.hidden_size // self.n_heads,
        )
        return self._apply_rotary(qkv, rotary_cos_sin)

    @staticmethod
    def _apply_rotary(
        qkv: Tensor,
        rotary_cos_sin: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply upstream RoPE without silently promoting Q/K under AMP.

        Untouched SemiCat keeps the cached RoPE tensors in float32. Its helper
        therefore promotes bf16 Q/K to float32 while V stays bf16, which is
        rejected by both FlexAttention and SDPA. Casting the cache at the block
        execution boundary is the same AMP fix used by official BD3-LM and does
        not add or alter model parameters.
        """
        cos, sin = rotary_cos_sin
        return split_and_apply_rotary_pos_emb(
            qkv,
            (cos.to(dtype=qkv.dtype), sin.to(dtype=qkv.dtype)),
        )

    def _doubled_qkv(
        self,
        layer: nn.Module,
        h: Tensor,
        rotary_cos_sin,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, tokens, _ = h.shape
        if tokens != 2 * self.length:
            raise ValueError(f"doubled stream must have length {2 * self.length}, got {tokens}")
        qkv = layer.attn_qkv(h).reshape(
            batch,
            tokens,
            3,
            self.n_heads,
            self.hidden_size // self.n_heads,
        )
        clean = self._apply_rotary(qkv[:, : self.length], rotary_cos_sin)
        noisy = self._apply_rotary(qkv[:, self.length :], rotary_cos_sin)
        return tuple(torch.cat((c, n), dim=1) for c, n in zip(clean, noisy))  # type: ignore[return-value]

    @staticmethod
    def _layer_modulation(layer: nn.Module, cond: Tensor) -> tuple[Tensor, ...]:
        return layer.adaLN_modulation(cond).chunk(6, dim=-1)

    @staticmethod
    def _finish_layer(
        layer: nn.Module,
        residual: Tensor,
        attention_output: Tensor,
        modulation: tuple[Tensor, ...],
    ) -> Tensor:
        _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        h = layer.dropout(layer.attn_out(attention_output) * gate_msa) + residual
        residual = h
        h = layer.norm2(h) * (1.0 + scale_mlp) + shift_mlp
        return layer.dropout(layer.mlp(h) * gate_mlp) + residual

    def _final_layer(self, h: Tensor, cond: Tensor) -> Tensor:
        h = self.output_layer.norm_final(h)
        shift, scale = self.output_layer.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.output_layer.linear(modulate_fused(h, shift, scale))

    @staticmethod
    def _sdpa(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        *,
        math_only: bool,
    ) -> Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if math_only:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
        else:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        return output.transpose(1, 2).reshape(
            query.shape[0],
            query.shape[2],
            query.shape[1] * query.shape[3],
        )

    def _normal_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        clean_only: bool,
        dense_mask: Tensor | None = None,
        force_math: bool = False,
    ) -> Tensor:
        use_flex = (
            not force_math
            and dense_mask is None
            and self.attention_backend == "flex"
            and FLEX_ATTENTION_AVAILABLE
            and query.is_cuda
        )
        if use_flex:
            mask = self._flex_mask(query.device, clean_only=clean_only)
            output = _compiled_flex_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                mask,  # type: ignore[arg-type]
            )
            return output.transpose(1, 2).reshape(
                query.shape[0],
                query.shape[1],
                self.hidden_size,
            )

        if (
            self.attention_backend == "flex"
            and not self._warned_flex_fallback
            and not force_math
            and dense_mask is None
            and (not FLEX_ATTENTION_AVAILABLE or not query.is_cuda)
        ):
            self._warned_flex_fallback = True
            print(
                "[BlockDIT] FlexAttention unavailable for this device; "
                "falling back to dense SDPA."
            )
        if dense_mask is None:
            dense_mask = (
                self._dense_clean_mask(query.device)
                if clean_only
                else block_causal_mask(self.length, self.block_size, query.device)
            )
        return self._sdpa(
            query,
            key,
            value,
            dense_mask,
            math_only=force_math,
        )

    # --------------------------------------------------------------- ordinary forward
    def forward(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        attn_mask: Tensor | None = None,
        jvp_attention: bool = False,
    ) -> Tensor:
        """Run one doubled-sequence pass and return logits for both streams.

        ``jvp_attention=True`` is a portable dense fallback used by CPU tests.
        Production self-distillation calls :meth:`forward_noisy_jvp` instead,
        which uses cached clean KV and the sparse custom-kernel path.
        """
        batch, tokens, vocab = x.shape
        if tokens != 2 * self.length or vocab != self.vocab_size:
            raise ValueError(
                f"expected x=(batch,{2 * self.length},{self.vocab_size}), got {tuple(x.shape)}"
            )
        cond = self._condition(s, t)
        h = self._embed(x, s, cond)
        rotary_cos_sin = self.rotary_emb()

        for layer in self.blocks:
            modulation = self._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = self._doubled_qkv(layer, h_norm, rotary_cos_sin)
            attention_output = self._normal_attention(
                query,
                key,
                value,
                clean_only=False,
                dense_mask=attn_mask,
                force_math=jvp_attention,
            )
            h = self._finish_layer(layer, residual, attention_output, modulation)
        return self._final_layer(h, cond)

    # ------------------------------------------------------------ optimized JVP forward
    def build_clean_kv_cache(self, clean: Tensor) -> CleanKVCache:
        """Compute clean-stream KV once, outside ``torch.func.jvp``.

        Gradients are retained: the noisy self-distillation loss can update the
        prefix representation through the cached keys and values.
        """
        batch, tokens, vocab = clean.shape
        if tokens != self.length or vocab != self.vocab_size:
            raise ValueError(
                f"expected clean=(batch,{self.length},{self.vocab_size}), got {tuple(clean.shape)}"
            )
        ones = torch.ones(batch, self.length, device=clean.device, dtype=clean.dtype)
        cond = self._condition(ones, ones)
        h = self._embed(clean, ones, cond)
        rotary_cos_sin = self.rotary_emb()
        keys: list[Tensor] = []
        values: list[Tensor] = []

        for layer in self.blocks:
            modulation = self._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = self._stream_qkv(layer, h_norm, rotary_cos_sin)
            keys.append(key)
            values.append(value)
            attention_output = self._normal_attention(
                query,
                key,
                value,
                clean_only=True,
            )
            h = self._finish_layer(layer, residual, attention_output, modulation)

        return CleanKVCache(
            keys=tuple(keys),
            values=tuple(values),
            batch_size=batch,
            length=self.length,
        )

    def _jvp_attention(
        self,
        query: Tensor,
        noisy_key: Tensor,
        noisy_value: Tensor,
        clean_key: Tensor,
        clean_value: Tensor,
    ) -> Tensor:
        """Sparse block attention used only from inside ``torch.func.jvp``."""
        use_triton = (
            self.jvp_attention_backend in {"auto", "triton"} and query.is_cuda
        )
        if self.jvp_attention_backend == "triton" and not query.is_cuda:
            raise RuntimeError("jvp_attention_backend='triton' requires CUDA")

        outputs: list[Tensor] = []
        for block_index in range(self.length // self.block_size):
            lo = block_index * self.block_size
            hi = lo + self.block_size
            block_query = query[:, lo:hi]
            block_key = torch.cat((clean_key[:, :lo], noisy_key[:, lo:hi]), dim=1)
            block_value = torch.cat((clean_value[:, :lo], noisy_value[:, lo:hi]), dim=1)
            if use_triton:
                # Match untouched Duo's custom-kernel call contract. Slicing a
                # block preserves the full-sequence batch stride; the Triton
                # reverse-over-forward kernel requires compact Q/K/V strides
                # (the mismatch is otherwise hidden at batch size one).
                block_output = safe_sdpa_jvp(
                    block_query.contiguous(),
                    block_key.contiguous(),
                    block_value.contiguous(),
                )
                block_output = block_output.reshape(
                    query.shape[0],
                    self.block_size,
                    self.hidden_size,
                )
            else:
                block_output = self._sdpa(
                    block_query,
                    block_key,
                    block_value,
                    mask=None,
                    math_only=True,
                )
            outputs.append(block_output)
        return torch.cat(outputs, dim=1)

    def forward_noisy_jvp(
        self,
        noisy: Tensor,
        s: Tensor,
        t: Tensor,
        clean_cache: CleanKVCache,
    ) -> Tensor:
        """Return noisy-stream logits using the custom JVP-compatible path.

        This method must be called from inside ``torch.func.jvp``. The clean
        cache itself must be built immediately beforehand, outside that transform.
        """
        batch, tokens, vocab = noisy.shape
        if (
            tokens != self.length
            or vocab != self.vocab_size
            or clean_cache.batch_size != batch
            or clean_cache.length != self.length
        ):
            raise ValueError("noisy input and clean cache are incompatible")
        cond = self._condition(s, t)
        h = self._embed(noisy, s, cond)
        rotary_cos_sin = self.rotary_emb()

        for layer_index, layer in enumerate(self.blocks):
            modulation = self._layer_modulation(layer, cond)
            shift_msa, scale_msa, *_ = modulation
            residual = h
            h_norm = modulate_fused(layer.norm1(h), shift_msa, scale_msa)
            query, key, value = self._stream_qkv(layer, h_norm, rotary_cos_sin)
            attention_output = self._jvp_attention(
                query,
                key,
                value,
                clean_cache.keys[layer_index],
                clean_cache.values[layer_index],
            )
            h = self._finish_layer(layer, residual, attention_output, modulation)
        return self._final_layer(h, cond)
