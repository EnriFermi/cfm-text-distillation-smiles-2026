"""Model checkpointing with an exact full-state checkpoint at train end."""

from __future__ import annotations

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ReliableModelCheckpoint(ModelCheckpoint):
    """Persist ``last.ckpt`` at the actual final optimizer step.

    The Lightning version used by this repository only writes ``last.ckpt`` in
    ``on_train_end`` when no earlier checkpoint has ever been written.  If a
    monitored best checkpoint was saved earlier, the existing ``last.ckpt`` can
    therefore remain stale even though training reaches ``max_steps``.

    This subclass uses ``_last_global_step_saved`` as the condition instead: an
    existing last checkpoint is reused only when it was written at the current
    global step.  Periodic checkpoint behavior remains unchanged.  Since
    ``save_weights_only`` defaults to false, optimizer and scheduler state are
    included unless a caller explicitly opts out.
    """

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.save_last:
            return

        current_step = int(trainer.global_step)
        last_saved_step = int(self._last_global_step_saved)
        if last_saved_step == current_step and self.last_model_path:
            rank_zero_info(
                "ReliableModelCheckpoint final cache hit: "
                f"global_step={current_step}, path={self.last_model_path}"
            )
            return

        monitor_candidates = self._monitor_candidates(trainer)
        rank_zero_info(
            "ReliableModelCheckpoint saving exact final full state: "
            f"previous_step={last_saved_step}, final_step={current_step}, "
            f"dirpath={self.dirpath}, save_weights_only={self.save_weights_only}"
        )

        # ``save_last='link'`` would otherwise link to the most recent periodic
        # checkpoint, which may be stale at an off-cadence final step.  Force a
        # real save for this one operation so last.ckpt contains current state.
        configured_save_last = self.save_last
        if configured_save_last == "link":
            self.save_last = True
        try:
            self._save_last_checkpoint(trainer, monitor_candidates)
        finally:
            self.save_last = configured_save_last

        rank_zero_info(
            "ReliableModelCheckpoint saved exact final full state: "
            f"global_step={current_step}, path={self.last_model_path}"
        )

