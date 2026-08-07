"""Atomically save and verify an immutable full-state checkpoint at the target step."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class VerifiedFinalCheckpoint(Callback):
    """Write a distinct final checkpoint and fail if its full state is invalid."""

    def __init__(
        self,
        dirpath: str,
        expected_global_step: int,
        filename: str = "step_{step:08d}_full_state.ckpt",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath).expanduser()
        self.expected_global_step = int(expected_global_step)
        self.filename = filename
        try:
            filename.format(step=self.expected_global_step)
        except (KeyError, ValueError) as error:
            raise ValueError("filename may only format step") from error

    @property
    def state_key(self) -> str:
        return (
            f"{type(self).__qualname__}[dirpath={self.dirpath},"
            f"expected_global_step={self.expected_global_step}]"
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
        rank_zero_info(
            "VerifiedFinalCheckpoint armed: "
            f"expected_global_step={self.expected_global_step}, dirpath={self.dirpath}"
        )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = int(trainer.global_step)
        if step != self.expected_global_step:
            raise RuntimeError(
                "Refusing to label an incomplete run as final: "
                f"expected global_step={self.expected_global_step}, found {step}"
            )
        if not trainer.is_global_zero:
            return

        filename = self.filename.format(step=step)
        checkpoint_path = self.dirpath / filename
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
        manifest_path = checkpoint_path.with_suffix(".verified.json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if checkpoint_path.exists():
            metadata = self._verify(checkpoint_path)
            rank_zero_info(
                "VerifiedFinalCheckpoint cache hit: "
                f"global_step={step}, path={checkpoint_path}"
            )
        else:
            rank_zero_info(
                "VerifiedFinalCheckpoint saving atomic full state: "
                f"global_step={step}, temporary_path={temporary_path}, "
                f"final_path={checkpoint_path}"
            )
            started = time.perf_counter()
            try:
                trainer.save_checkpoint(temporary_path, weights_only=False)
                os.replace(temporary_path, checkpoint_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            metadata = self._verify(checkpoint_path)
            metadata["save_and_verify_seconds"] = time.perf_counter() - started

        manifest_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rank_zero_info(
            "VerifiedFinalCheckpoint verified: "
            f"global_step={step}, optimizer_states={metadata['optimizer_states']}, "
            f"lr_schedulers={metadata['lr_schedulers']}, "
            f"size_bytes={metadata['size_bytes']}, sha256={metadata['sha256']}, "
            f"path={checkpoint_path}, manifest={manifest_path}"
        )

    def _verify(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        actual_step = int(checkpoint.get("global_step", -1))
        optimizer_states = checkpoint.get("optimizer_states", [])
        lr_schedulers = checkpoint.get("lr_schedulers", [])
        state_dict = checkpoint.get("state_dict", {})

        errors = []
        if actual_step != self.expected_global_step:
            errors.append(
                f"global_step={actual_step}, expected={self.expected_global_step}"
            )
        if not optimizer_states:
            errors.append("optimizer_states is empty")
        if not lr_schedulers:
            errors.append("lr_schedulers is empty")
        if not state_dict:
            errors.append("state_dict is empty")
        if errors:
            raise RuntimeError(
                f"Final checkpoint verification failed for {checkpoint_path}: "
                + "; ".join(errors)
            )

        return {
            "checkpoint": str(checkpoint_path.resolve()),
            "global_step": actual_step,
            "epoch": int(checkpoint.get("epoch", -1)),
            "optimizer_states": len(optimizer_states),
            "optimizer_state_entries": [
                len(optimizer.get("state", {})) for optimizer in optimizer_states
            ],
            "lr_schedulers": len(lr_schedulers),
            "state_dict_entries": len(state_dict),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": self._sha256(checkpoint_path),
            "verified": True,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
