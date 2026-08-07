import pytest
import torch

from semicat.train import validate_resume_checkpoint


def test_resume_checkpoint_step_guard_accepts_exact_step(tmp_path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"global_step": 200000}, checkpoint)

    validate_resume_checkpoint(str(checkpoint), 200000)


def test_resume_checkpoint_step_guard_rejects_best_or_stale_step(tmp_path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"global_step": 144001}, checkpoint)

    with pytest.raises(RuntimeError, match=r"expected global_step=200000.*found global_step=144001"):
        validate_resume_checkpoint(str(checkpoint), 200000)

