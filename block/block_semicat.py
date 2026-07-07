"""Training-time BCFM (M3): block-factorized VFM + ECLD over the BD3-LM doubled sequence.

Realizes paper §Methods Eq. (total): for each block ``b`` (conditioned on the clean prefix
``x^{(<b)}``) a mean-field flow-matching loss plus an Endpoint-Consistent Lagrangian
Distillation term, all obtained in **one** masked forward over ``[x_clean ; z_noisy]``.

The per-block ECLD mirrors ``semicat.models.semicat.SemicatModule.sd_model_step`` ("ecld"
branch) exactly — same JVP-of-the-denoiser construction — but with per-token times and the
block-causal mask, so a noisy block's targets read only its own state and the clean prefix.
Sampling reuses ``block.sampling.block_causal_sample``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from semicat.models.textsemicat import TextSemicatModule


class BlockSemicatModule(TextSemicatModule):
    """:param block_size: block length B. :param ecld_weight: λ on the ECLD term.
    :param time_eps: upper-time margin, ``t ~ U(0, 1-time_eps)``."""

    def __init__(self, *args, block_size: int = 16, ecld_weight: float = 1.0,
                 time_eps: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.ecld_weight = ecld_weight
        self.time_eps = time_eps
        self.save_hyperparameters("block_size", "ecld_weight", "time_eps")
        self._mask = None

    # ------------------------------------------------------------------ helpers
    def _mask_for(self, device) -> Tensor:
        if self._mask is None or self._mask.device != device:
            from block.mask import block_causal_mask
            self._mask = block_causal_mask(self.in_shape[0], self.block_size, device)
        return self._mask

    def _block_times(self, batch: int, n_blocks: int, device) -> tuple[Tensor, Tensor]:
        """Per-block ``s_b < t_b`` (each (batch, n_blocks)), then use ``_expand``."""
        t = torch.rand(batch, n_blocks, device=device) * (1.0 - self.time_eps)
        s = torch.rand(batch, n_blocks, device=device) * t
        return s, t

    def _expand(self, per_block: Tensor) -> Tensor:
        """(batch, n_blocks) -> (batch, L) by repeating each block ``block_size`` times."""
        return per_block.repeat_interleave(self.block_size, dim=1)

    def _noisy_logits(self, clean: Tensor, z: Tensor, s_tok: Tensor, t_tok: Tensor) -> Tensor:
        """Run the masked doubled-sequence forward, return the noisy-half logits."""
        L = self.in_shape[0]
        ones = torch.ones_like(s_tok)  # clean stream sits at s=t=1
        x = torch.cat([clean, z], dim=1)
        s = torch.cat([ones, s_tok], dim=1)
        t = torch.cat([ones, t_tok], dim=1)
        return self.net(x, s, t, self._mask_for(z.device))[:, L:]

    # -------------------------------------------------------------------- losses
    def _block_vfm(self, x0: Tensor, x1: Tensor, delta: Tensor) -> Tensor:
        """Mean-field CE recovering each clean block from its noised copy at s=t=t_b."""
        batch, L = x1.shape
        _, t = self._block_times(batch, L // self.block_size, x1.device)
        t_tok = self._expand(t)
        z = self._interpolate(x0, x1, t_tok[..., None])
        pred = self._noisy_logits(delta, z, t_tok, t_tok)
        return F.cross_entropy(pred.transpose(-1, 1), x1,
                               label_smoothing=self.hparams.label_smoothing)

    def _block_ecld(self, x0: Tensor, x1: Tensor, delta: Tensor) -> Tensor:
        """Per-block ECLD (paper Eq. L_ECLD), same construction as SemicatModule."""
        batch, L = x1.shape
        s, t = self._block_times(batch, L // self.block_size, x1.device)
        s_tok, t_tok = self._expand(s), self._expand(t)
        zs = self._interpolate(x0, x1, s_tok[..., None])

        # q_{s,t}(z_s ; c) and its d/dt via forward-mode AD (masked attention is jvp-safe)
        must, dmu = torch.func.jvp(
            lambda _t: self._noisy_logits(delta, zs, s_tok, _t).softmax(dim=-1),
            (t_tok,), (torch.ones_like(t_tok),),
        )

        with torch.no_grad():
            gamma = (t_tok - s_tok) / (1.0 - s_tok + 1e-8)          # (batch, L)
            xst = zs + gamma[..., None] * (must - zs)               # flow map X_{s,t}(z_s)
            teacher = self._noisy_logits(delta, xst, t_tok, t_tok).softmax(dim=-1)

        div = -(teacher * must.clamp_min(1e-9).log()).sum(dim=-1).mean()
        energy = (gamma[..., None] * dmu).pow(2).sum(dim=-1).mean()
        return div + energy

    def model_step(self, batch):
        x1 = batch["input_ids"] if isinstance(batch, dict) else batch  # (batch, L) ids
        K = self.in_shape[-1]
        x0 = self.prior((x1.size(0), *self.in_shape), device=x1.device)
        delta = F.one_hot(x1, K).to(x0.dtype)

        sd_split = int(self.hparams.sd_prop * x1.size(0))
        vf = self._block_vfm(x0[sd_split:], x1[sd_split:], delta[sd_split:])
        if sd_split == 0:
            return vf, None
        sd = self.ecld_weight * self._block_ecld(x0[:sd_split], x1[:sd_split], delta[:sd_split])
        return vf, sd

    # ------------------------------------------------------------------ sampling
    @torch.inference_mode()
    def sample_flow_map_batch(self, batch_size: int, sampling_steps: int, x0=None) -> Tensor:
        """Block-causal sampler, returned as one-hot so the inherited text eval loop works."""
        from block.sampling import block_causal_sample
        toks = block_causal_sample(self, self.block_size, sampling_steps,
                                   batch_size=batch_size, length=self.in_shape[0])
        return F.one_hot(toks, self.in_shape[-1]).float()
