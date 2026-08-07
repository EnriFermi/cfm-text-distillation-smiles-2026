"""Runtime visibility and safe performance switches for long training runs."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class RuntimeDiagnostics(Callback):
    """Report the resolved runtime and enable fused AdamW after state restore."""

    def __init__(
        self,
        run_label: str,
        output_dir: str,
        seed: int,
        force_foreach_adamw: bool = True,
    ) -> None:
        super().__init__()
        self.run_label = run_label
        self.output_dir = Path(output_dir).expanduser()
        self.seed = int(seed)
        self.force_foreach_adamw = bool(force_foreach_adamw)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device = trainer.strategy.root_device
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        parameter = next(pl_module.parameters())
        net = getattr(pl_module, "net", None)
        rank_zero_info(
            "Runtime start: "
            f"label={self.run_label}, output_dir={self.output_dir}, seed={self.seed}, "
            f"device={device}, gpu={gpu}, precision={trainer.precision}, "
            f"parameter_dtype={parameter.dtype}, matmul_precision="
            f"{torch.get_float32_matmul_precision()}, "
            f"jvp_attention_backend={getattr(net, 'jvp_attention_backend', 'unknown')}, "
            f"max_steps={trainer.max_steps}, accumulate_grad_batches="
            f"{trainer.accumulate_grad_batches}, val_check_interval="
            f"{trainer.val_check_interval}"
        )

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        descriptions = []
        for optimizer in trainer.optimizers:
            if self.force_foreach_adamw and isinstance(optimizer, torch.optim.AdamW):
                # Lightning's AMP plugin deliberately rejects gradient clipping
                # for fused AdamW. CUDA foreach keeps clip=1.0 (part of the source
                # recipe) while batching the elementwise optimizer operations.
                optimizer.defaults["fused"] = False
                optimizer.defaults["foreach"] = True
                for group in optimizer.param_groups:
                    group["fused"] = False
                    group["foreach"] = True

            state_steps = []
            for state in optimizer.state.values():
                step = state.get("step")
                if isinstance(step, torch.Tensor) and step.numel() == 1:
                    state_steps.append(float(step.detach().cpu()))
                elif isinstance(step, (int, float)):
                    state_steps.append(float(step))
            finite_steps = [step for step in state_steps if math.isfinite(step)]
            descriptions.append(
                f"{type(optimizer).__name__}(lr={[g['lr'] for g in optimizer.param_groups]}, "
                f"fused={[g.get('fused') for g in optimizer.param_groups]}, "
                f"foreach={[g.get('foreach') for g in optimizer.param_groups]}, "
                f"state_entries={len(optimizer.state)}, "
                f"state_step_range="
                f"{(min(finite_steps), max(finite_steps)) if finite_steps else None})"
            )
        scheduler_descriptions = [
            f"{type(item.scheduler).__name__}(last_epoch="
            f"{getattr(item.scheduler, 'last_epoch', None)}, interval={item.interval})"
            for item in trainer.lr_scheduler_configs
        ]
        rank_zero_info(
            "Runtime restored: "
            f"global_step={trainer.global_step}, epoch={trainer.current_epoch}, "
            f"optimizers={descriptions}, schedulers={scheduler_descriptions}"
        )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            rank_zero_info(
                f"Pipeline stage=validation_start label={self.run_label} "
                f"step={trainer.global_step} epoch={trainer.current_epoch}"
            )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("val/loss")
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = float(value.detach().cpu())
        rank_zero_info(
            f"Pipeline stage=validation_done label={self.run_label} "
            f"step={trainer.global_step} epoch={trainer.current_epoch} val_loss={value}"
        )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        rank_zero_info(
            f"Runtime complete: label={self.run_label}, global_step={trainer.global_step}, "
            f"output_dir={self.output_dir}, callback_metrics={dict(trainer.callback_metrics)}"
        )
