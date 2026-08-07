from types import MethodType, SimpleNamespace

import torch

from semicat.callbacks.reliable_model_checkpoint import ReliableModelCheckpoint


def test_regression_stale_last_144001_is_resaved_at_final_200000(tmp_path) -> None:
    callback = ReliableModelCheckpoint(
        dirpath=tmp_path,
        filename="best",
        monitor="val/loss",
        save_last=True,
        save_top_k=1,
        save_weights_only=False,
    )
    callback._last_global_step_saved = 144001
    callback._last_checkpoint_saved = str(tmp_path / "best.ckpt")
    callback.last_model_path = str(tmp_path / "last.ckpt")

    trainer = SimpleNamespace(
        global_step=200000,
        current_epoch=15,
        callback_metrics={"val/loss": torch.tensor(3.95)},
    )
    saved_steps: list[int] = []

    def save_last(self, trainer, monitor_candidates) -> None:
        assert abs(float(monitor_candidates["val/loss"]) - 3.95) < 1e-6
        saved_steps.append(int(trainer.global_step))
        self._last_global_step_saved = int(trainer.global_step)

    callback._save_last_checkpoint = MethodType(save_last, callback)
    callback.on_train_end(trainer, None)

    assert saved_steps == [200000]
    assert callback._last_global_step_saved == 200000


def test_exact_periodic_final_checkpoint_is_not_written_twice(tmp_path) -> None:
    callback = ReliableModelCheckpoint(
        dirpath=tmp_path,
        monitor=None,
        save_last=True,
        every_n_train_steps=10000,
    )
    callback._last_global_step_saved = 400000
    callback.last_model_path = str(tmp_path / "last.ckpt")
    trainer = SimpleNamespace(global_step=400000)

    def unexpected_save(self, trainer, monitor_candidates) -> None:
        raise AssertionError("An exact final checkpoint must not be written twice")

    callback._save_last_checkpoint = MethodType(unexpected_save, callback)
    callback.on_train_end(trainer, None)
