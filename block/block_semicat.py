"""Block-wise generalization of the untouched upstream SemiCat objectives.

Only the formulation-specific pieces differ from ``SemicatModule``:

* one time pair per block;
* a doubled clean/noisy block-causal model forward;
* strict initialization of that model from a trained full-sequence CFM.

VFM, CFM/Lagrangian (upstream ``sd_type=lag``), and ECLD formulas, batch
splitting, reductions, interpolation, prior, optimizer setup, and the Text8
dataloader remain upstream-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from block.block_dit import BlockDIT
from block.checkpoint import (
    CFMInitializationReport,
    initialize_blockdit_from_cfm,
    write_initialization_report,
)
from semicat.models.textsemicat import TextSemicatModule


class BlockSemicatModule(TextSemicatModule):
    """Fine-tune a trained CFM as a genuine block-conditional flow map."""

    def __init__(
        self,
        *args,
        block_size: int = 16,
        init_from_cfm_ckpt: str | None = None,
        expected_source_sd_type: str | None = "lag",
        time_eps: float = 0.0,
        sd_type: Literal["cfm", "lag", "ecld", "semi"] = "lag",
        **kwargs,
    ):
        if sd_type == "cfm":
            sd_type = "lag"
        super().__init__(*args, sd_type=sd_type, **kwargs)
        if not isinstance(self.net, BlockDIT):
            raise TypeError(f"BlockSemicatModule requires BlockDIT, got {type(self.net).__name__}")
        if self.in_shape[0] % block_size:
            raise ValueError(
                f"sequence length {self.in_shape[0]} is not divisible by block_size {block_size}"
            )
        if self.net.block_size != block_size:
            raise ValueError(
                f"module block_size={block_size} does not match net.block_size={self.net.block_size}"
            )
        if not 0.0 <= time_eps < 1.0:
            raise ValueError("time_eps must lie in [0, 1)")

        self.block_size = int(block_size)
        self.init_from_cfm_ckpt = init_from_cfm_ckpt
        self.expected_source_sd_type = expected_source_sd_type
        self.time_eps = float(time_eps)
        self._cfm_initialization_report: CFMInitializationReport | None = None
        self.save_hyperparameters(
            "block_size",
            "init_from_cfm_ckpt",
            "expected_source_sd_type",
            "time_eps",
        )

    # ---------------------------------------------------------------------- times
    def _block_times(
        self,
        batch_size: int,
        n_blocks: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        t = self._block_t(batch_size, n_blocks, device)
        s = torch.rand_like(t) * t
        return s, t

    def _block_t(
        self,
        batch_size: int,
        n_blocks: int,
        device: torch.device,
    ) -> Tensor:
        """Draw exactly the upstream VFM random variable, once per block."""
        t = torch.rand(batch_size, n_blocks, device=device)
        if self.time_eps:
            t = t * (1.0 - self.time_eps)
        return t

    def _expand_block_times(self, times: Tensor) -> Tensor:
        return times.repeat_interleave(self.block_size, dim=1)

    # ---------------------------------------------------------------- model calls
    def _combined_noisy_logits(
        self,
        clean: Tensor,
        noisy: Tensor,
        s: Tensor,
        t: Tensor,
    ) -> Tensor:
        """One official BD3-LM-style masked forward, returning the noisy half."""
        length = self.in_shape[0]
        ones = torch.ones_like(s)
        doubled = torch.cat((clean, noisy), dim=1)
        doubled_s = torch.cat((ones, s), dim=1)
        doubled_t = torch.cat((ones, t), dim=1)
        return self.net(doubled, doubled_s, doubled_t)[:, length:]

    def _jvp_noisy_logits(
        self,
        noisy: Tensor,
        s: Tensor,
        t: Tensor,
        clean_cache,
    ) -> Tensor:
        return self.net.forward_noisy_jvp(noisy, s, t, clean_cache)

    # --------------------------------------------------------------------- losses
    def vfm_model_step(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Upstream VFM CE with the single sample time generalized per block."""
        batch_size, length = x1.shape
        t = self._block_t(
            batch_size,
            length // self.block_size,
            x1.device,
        )
        t = self._expand_block_times(t)
        xt = self._interpolate(x0, x1, t[..., None])
        clean = F.one_hot(x1, self.in_shape[-1]).to(x0.dtype)
        prediction = self._combined_noisy_logits(clean, xt, t, t)
        return F.cross_entropy(
            prediction.transpose(-1, 1),
            x1,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _lag_model_step(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Exact upstream CFM/Lagrangian loss with per-block ``(s_b,t_b)``."""
        batch_size, length = x1.shape
        s, t = self._block_times(
            batch_size,
            length // self.block_size,
            x1.device,
        )
        s = self._expand_block_times(s)
        t = self._expand_block_times(t)
        xs = self._interpolate(x0, x1, s[..., None])
        clean = F.one_hot(x1, self.in_shape[-1]).to(x0.dtype)

        # FlexAttention cannot be transformed by torch.func.jvp. Compute the
        # clean stream once outside JVP, retaining its reverse-mode graph.
        clean_cache = self.net.build_clean_kv_cache(clean)

        def block_xst(target_time: Tensor) -> Tensor:
            prediction = self._jvp_noisy_logits(
                xs,
                s,
                target_time,
                clean_cache,
            ).softmax(dim=-1)
            dt = (target_time - s) / (1.0 - s + 1e-8)
            return xs + dt[..., None] * (prediction - xs)

        xst, dv = torch.func.jvp(
            block_xst,
            primals=(t,),
            tangents=(torch.ones_like(t),),
        )
        xst = xst.detach()
        with torch.no_grad():
            dv_target = self._combined_noisy_logits(clean, xst, t, t).softmax(dim=-1)
        return (
            (1.0 - t[..., None]) * dv - dv_target + xst
        ).pow(2).sum(dim=-1).mean()

    def _ecld_model_step(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Exact upstream ECLD CE+TD reduction with per-block times."""
        batch_size, length = x1.shape
        s, t = self._block_times(
            batch_size,
            length // self.block_size,
            x1.device,
        )
        s = self._expand_block_times(s)
        t = self._expand_block_times(t)
        xs = self._interpolate(x0, x1, s[..., None])
        clean = F.one_hot(x1, self.in_shape[-1]).to(x0.dtype)
        clean_cache = self.net.build_clean_kv_cache(clean)

        must, dmu = torch.func.jvp(
            lambda target_time: self._jvp_noisy_logits(
                xs,
                s,
                target_time,
                clean_cache,
            ).softmax(dim=-1),
            primals=(t,),
            tangents=(torch.ones_like(t),),
        )
        with torch.no_grad():
            gamma = (t - s) / (1.0 - s + 1e-8)
            xst = xs + gamma[..., None] * (must - xs)
            mutt = self._combined_noisy_logits(clean, xst, t, t).softmax(dim=-1)

        div = -(mutt * must.log()).sum(dim=-1).mean()
        # Match upstream exactly: sum over sequence and vocabulary, mean batch.
        energy = (gamma[..., None] * dmu).pow(2).sum(dim=(1, 2)).mean()
        return div + energy

    def sd_model_step(self, x0: Tensor, x1: Tensor) -> Tensor:
        if self.hparams.sd_type == "lag":
            return self._lag_model_step(x0, x1)
        if self.hparams.sd_type == "ecld":
            return self._ecld_model_step(x0, x1)
        if self.hparams.sd_type == "semi":
            raise NotImplementedError("not implemented yet")
        raise ValueError(f"unknown sd_type: {self.hparams.sd_type!r}")

    def model_step(
        self,
        batch: Tensor | dict[str, Tensor],
    ) -> tuple[Tensor, Tensor | None]:
        """Preserve upstream prior sampling and VFM/SD batch partition exactly."""
        x1 = batch["input_ids"] if isinstance(batch, dict) else batch
        x0 = self.prior((x1.size(0), *self.in_shape), device=x1.device)
        sd_split = int(self.hparams.sd_prop * x1.size(0))
        vf_loss = self.vfm_model_step(x0[sd_split:], x1[sd_split:])
        if sd_split == 0:
            return vf_loss, None
        return vf_loss, self.sd_model_step(x0[:sd_split], x1[:sd_split])

    # --------------------------------------------------------- checkpoint lifecycle
    def initialize_from_cfm_checkpoint(self) -> CFMInitializationReport:
        if not self.init_from_cfm_ckpt:
            raise ValueError("init_from_cfm_ckpt is not configured")
        if self._cfm_initialization_report is None:
            self._cfm_initialization_report = initialize_blockdit_from_cfm(
                self.net,
                self.init_from_cfm_ckpt,
                expected_in_shape=self.in_shape,
                expected_source_sd_type=self.expected_source_sd_type,
            )
        return self._cfm_initialization_report

    def on_fit_start(self) -> None:
        resume_path = getattr(self.trainer, "ckpt_path", None)
        report_path: Path | None = None
        if resume_path:
            self.print(
                "[BlockSemicat] stage=initialization mode=resume "
                f"checkpoint={resume_path}; skipping CFM initialization"
            )
        elif self.init_from_cfm_ckpt:
            report = self.initialize_from_cfm_checkpoint()
            report_path = write_initialization_report(
                report,
                self.trainer.default_root_dir,
            )
            self.print(
                "[BlockSemicat] stage=initialization mode=cfm_finetune "
                f"checkpoint={report.checkpoint} source_step={report.global_step} "
                f"source_loss={report.source_sd_type} tensors={report.loaded_tensors}"
            )
        else:
            self.print("[BlockSemicat] stage=initialization mode=scratch")

        self.print(
            "[BlockSemicat] stage=model_ready "
            f"device={self.device} parameter_dtype={next(self.parameters()).dtype} "
            f"block_size={self.block_size} loss={self.hparams.sd_type} "
            f"attention={self.net.attention_backend} "
            f"jvp_attention={self.net.jvp_attention_backend} "
            f"init_report={report_path}"
        )
        super().on_fit_start()

    # -------------------------------------------------------------------- sampling
    @torch.inference_mode()
    def sample_flow_map_batch(
        self,
        batch_size: int,
        sampling_steps: int,
        x0=None,
    ) -> Tensor:
        if x0 is not None:
            raise ValueError("block sampler does not accept a full-sequence x0")
        from block.sampling import block_causal_sample

        tokens = block_causal_sample(
            self,
            self.block_size,
            sampling_steps,
            batch_size=batch_size,
            length=self.in_shape[0],
        )
        return F.one_hot(tokens, self.in_shape[-1]).float()

    def _log_strings(self, title: str, xs: list[str]) -> None:
        """Keep upstream core untouched while making block runs Comet-safe."""
        if len(xs) > 64:
            xs = xs[:64]
        experiment = getattr(self.logger, "experiment", None)
        if self.logger.__class__.__name__ == "WandbLogger":
            import wandb

            table = wandb.Table(columns=["Text"])
            for text in xs:
                table.add_data(text)
            experiment.log({title: table}, commit=False)
        elif hasattr(experiment, "log_text"):
            experiment.log_text("\n".join([title, *xs]))
        print(f"{title}: {xs}")
