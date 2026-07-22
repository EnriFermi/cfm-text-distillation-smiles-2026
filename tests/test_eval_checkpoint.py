from types import SimpleNamespace

import torch
from lightning.pytorch.trainer.states import TrainerFn

from semicat.callbacks.eval_checkpoint import EvalCheckpoint


class _TrainerStub:
    def __init__(self, step: int = 10, epoch: int = 2) -> None:
        self.global_step = step
        self.current_epoch = epoch
        self.sanity_checking = False
        self.state = SimpleNamespace(fn=TrainerFn.FITTING)
        self.callback_metrics = {
            "val/loss": torch.tensor(1.25),
            "val/vector": torch.tensor([1.0, 2.0]),
            "train/loss": 2.5,
        }
        self.is_global_zero = True
        self.saved: list[tuple[str, bool]] = []

    def save_checkpoint(self, path, weights_only: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"full checkpoint")
        self.saved.append((str(path), weights_only))


def test_saves_distinct_full_checkpoint_after_each_eval(tmp_path) -> None:
    callback = EvalCheckpoint(dirpath=str(tmp_path))
    trainer = _TrainerStub()

    callback.on_fit_start(trainer, None)
    callback.on_validation_end(trainer, None)

    expected = tmp_path / "eval-step=00000010-epoch=0002.ckpt"
    assert trainer.saved == [(str(expected), False)]
    assert expected.read_bytes() == b"full checkpoint"
    metadata = expected.with_suffix(".json").read_text()
    assert '"global_step": 10' in metadata
    assert '"val/loss": 1.25' in metadata
    assert "val/vector" not in metadata

    # Duplicate validation at the same model state must not overwrite the file.
    callback.on_validation_end(trainer, None)
    assert len(trainer.saved) == 1

    trainer.global_step = 20
    trainer.current_epoch = 3
    callback.on_validation_end(trainer, None)
    assert len(trainer.saved) == 2
    assert trainer.saved[-1][0].endswith("eval-step=00000020-epoch=0003.ckpt")


def test_every_n_evals_and_state_restore(tmp_path) -> None:
    callback = EvalCheckpoint(dirpath=str(tmp_path), every_n_evals=2)
    trainer = _TrainerStub()

    callback.on_validation_end(trainer, None)
    assert trainer.saved == []

    trainer.global_step = 20
    callback.on_validation_end(trainer, None)
    assert len(trainer.saved) == 1

    restored = EvalCheckpoint(dirpath=str(tmp_path), every_n_evals=2)
    restored.load_state_dict(callback.state_dict())
    assert restored.state_dict() == callback.state_dict()

    # The restored callback recognizes the already-saved global step.
    restored.on_validation_end(trainer, None)
    assert len(trainer.saved) == 1


def test_skips_sanity_standalone_validation_and_step_zero(tmp_path) -> None:
    callback = EvalCheckpoint(dirpath=str(tmp_path))
    trainer = _TrainerStub(step=0)

    callback.on_validation_end(trainer, None)
    trainer.global_step = 10
    trainer.sanity_checking = True
    callback.on_validation_end(trainer, None)
    trainer.sanity_checking = False
    trainer.state.fn = TrainerFn.VALIDATING
    callback.on_validation_end(trainer, None)

    assert trainer.saved == []


def test_rejects_invalid_configuration(tmp_path) -> None:
    try:
        EvalCheckpoint(dirpath=str(tmp_path), every_n_evals=0)
    except ValueError as error:
        assert "every_n_evals" in str(error)
    else:
        raise AssertionError("Expected invalid every_n_evals to fail")

    try:
        EvalCheckpoint(dirpath=str(tmp_path), filename="{unknown}")
    except ValueError as error:
        assert "filename" in str(error)
    else:
        raise AssertionError("Expected invalid filename template to fail")
