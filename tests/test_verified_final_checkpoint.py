import json
from pathlib import Path

import torch

from semicat.callbacks.verified_final_checkpoint import VerifiedFinalCheckpoint


class _TrainerStub:
    def __init__(self, step: int) -> None:
        self.global_step = step
        self.is_global_zero = True

    def save_checkpoint(self, path: Path, weights_only: bool = False) -> None:
        assert weights_only is False
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": 3,
                "state_dict": {"weight": torch.ones(2)},
                "optimizer_states": [{"state": {0: {"step": self.global_step}}}],
                "lr_schedulers": [{"last_epoch": self.global_step}],
            },
            path,
        )


def test_saves_and_verifies_atomic_full_state(tmp_path) -> None:
    callback = VerifiedFinalCheckpoint(
        dirpath=str(tmp_path), expected_global_step=200000
    )
    trainer = _TrainerStub(step=200000)

    callback.on_fit_start(trainer, None)
    callback.on_train_end(trainer, None)

    checkpoint_path = tmp_path / "step_00200000_full_state.ckpt"
    manifest_path = tmp_path / "step_00200000_full_state.verified.json"
    assert checkpoint_path.is_file()
    assert not checkpoint_path.with_suffix(".ckpt.part").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["verified"] is True
    assert manifest["global_step"] == 200000
    assert manifest["optimizer_state_entries"] == [1]
    assert len(manifest["sha256"]) == 64


def test_refuses_incomplete_final_step(tmp_path) -> None:
    callback = VerifiedFinalCheckpoint(
        dirpath=str(tmp_path), expected_global_step=200000
    )
    trainer = _TrainerStub(step=199999)

    try:
        callback.on_train_end(trainer, None)
    except RuntimeError as error:
        assert "expected global_step=200000" in str(error)
    else:
        raise AssertionError("Incomplete training must not be saved as final")
