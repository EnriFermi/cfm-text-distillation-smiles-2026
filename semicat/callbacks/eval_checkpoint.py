"""Save a full training checkpoint after completed validation runs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class EvalCheckpoint(Callback):
    """Save a distinct full-state checkpoint after every N validation events.

    Unlike ``ModelCheckpoint(every_n_train_steps=...)``, this callback is tied
    directly to ``on_validation_end``. This matters when ``val_check_interval``
    counts training batches while ``global_step`` counts optimizer updates.
    """

    def __init__(
        self,
        dirpath: str,
        filename: str = "eval-step={step:08d}-epoch={epoch:04d}",
        every_n_evals: int = 1,
        save_weights_only: bool = False,
        skip_step_zero: bool = True,
    ) -> None:
        super().__init__()
        if every_n_evals < 1:
            raise ValueError("every_n_evals must be at least 1")

        # Fail at startup instead of after a long validation if the template is invalid.
        try:
            filename.format(step=0, epoch=0, eval_index=0)
        except (KeyError, ValueError) as error:
            raise ValueError(
                "filename may only format step, epoch, and eval_index"
            ) from error

        self.dirpath = Path(dirpath).expanduser()
        self.filename = filename
        self.every_n_evals = every_n_evals
        self.save_weights_only = save_weights_only
        self.skip_step_zero = skip_step_zero

        self._validation_events_seen = 0
        self._last_saved_global_step: int | None = None
        self._last_checkpoint_path: str | None = None

    @property
    def state_key(self) -> str:
        return (
            f"{type(self).__qualname__}[dirpath={self.dirpath},"
            f"every_n_evals={self.every_n_evals}]"
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        rank_zero_info(
            "EvalCheckpoint enabled: "
            f"dirpath={self.dirpath}, every_n_evals={self.every_n_evals}, "
            f"save_weights_only={self.save_weights_only}"
        )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return

        step = int(trainer.global_step)
        if self.skip_step_zero and step == 0:
            return
        if step == self._last_saved_global_step:
            return

        self._validation_events_seen += 1
        if self._validation_events_seen % self.every_n_evals != 0:
            return

        epoch = int(trainer.current_epoch)
        filename = self.filename.format(
            step=step,
            epoch=epoch,
            eval_index=self._validation_events_seen,
        )
        if not filename.endswith(".ckpt"):
            filename += ".ckpt"
        checkpoint_path = self.dirpath / filename

        if checkpoint_path.exists():
            self._last_saved_global_step = step
            self._last_checkpoint_path = str(checkpoint_path)
            rank_zero_info(
                f"EvalCheckpoint cache hit: reusing existing checkpoint {checkpoint_path}"
            )
            return

        metrics = self._scalar_metrics(trainer.callback_metrics)
        metric_context = ", ".join(
            f"{key}={value:.6g}"
            for key, value in sorted(metrics.items())
            if key.startswith("val/")
        )
        rank_zero_info(
            f"EvalCheckpoint saving: eval={self._validation_events_seen}, "
            f"epoch={epoch}, step={step}, path={checkpoint_path}"
            + (f", {metric_context}" if metric_context else "")
        )

        previous_step = self._last_saved_global_step
        previous_path = self._last_checkpoint_path
        self._last_saved_global_step = step
        self._last_checkpoint_path = str(checkpoint_path)
        started = time.perf_counter()
        try:
            trainer.save_checkpoint(
                checkpoint_path,
                weights_only=self.save_weights_only,
            )
        except Exception:
            self._last_saved_global_step = previous_step
            self._last_checkpoint_path = previous_path
            raise

        elapsed = time.perf_counter() - started
        size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else None
        if trainer.is_global_zero:
            metadata = {
                "checkpoint": str(checkpoint_path),
                "eval_index": self._validation_events_seen,
                "epoch": epoch,
                "global_step": step,
                "save_weights_only": self.save_weights_only,
                "size_bytes": size_bytes,
                "save_seconds": elapsed,
                "metrics": metrics,
            }
            checkpoint_path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )

        rank_zero_info(
            f"EvalCheckpoint saved: path={checkpoint_path}, "
            f"size_bytes={size_bytes}, seconds={elapsed:.3f}"
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        rank_zero_info(
            "EvalCheckpoint complete: "
            f"validation_events={self._validation_events_seen}, "
            f"last_checkpoint={self._last_checkpoint_path}"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "validation_events_seen": self._validation_events_seen,
            "last_saved_global_step": self._last_saved_global_step,
            "last_checkpoint_path": self._last_checkpoint_path,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._validation_events_seen = int(state_dict.get("validation_events_seen", 0))
        self._last_saved_global_step = state_dict.get("last_saved_global_step")
        self._last_checkpoint_path = state_dict.get("last_checkpoint_path")

    @staticmethod
    def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            if isinstance(value, (int, float)):
                scalar = float(value)
                if math.isfinite(scalar):
                    result[str(key)] = scalar
        return result
