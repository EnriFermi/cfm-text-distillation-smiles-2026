from functools import partial

import pytest
import torch
from omegaconf import OmegaConf

from block.block_dit import BlockDIT
from block.block_semicat import BlockSemicatModule
from block.checkpoint import write_initialization_report
from semicat.net.duo import DIT


def _kwargs():
    return dict(
        vocab_size=5,
        hidden_size=16,
        cond_dim=8,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
        length=8,
        embed_type="naive",
    )


def _checkpoint(path, *, sd_type="lag"):
    duo = DIT(**_kwargs())
    with torch.no_grad():
        for parameter in duo.parameters():
            parameter.normal_(std=0.05)
    torch.save(
        {
            "epoch": 12,
            "global_step": 345,
            "state_dict": {
                f"net.{key}": value for key, value in duo.state_dict().items()
            },
            "hyper_parameters": {
                "sd_type": sd_type,
                "in_shape": [8, 5],
            },
            "datamodule_hyper_parameters": {
                "train_val_test_split": OmegaConf.create([100, 20, 20]),
                "k": 8,
                "batch_size": 4,
            },
        },
        path,
    )
    return duo


def _module(checkpoint_path):
    net = BlockDIT(
        **_kwargs(),
        block_size=2,
        attention_backend="sdpa",
        jvp_attention_backend="math",
    )
    return BlockSemicatModule(
        net=net,
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        scheduler=None,
        in_shape=(8, 5),
        prior_type="gaussian",
        sd_prop=0.25,
        sd_type="lag",
        block_size=2,
        init_from_cfm_ckpt=str(checkpoint_path),
        expected_source_sd_type="lag",
        calc_nll=False,
    )


def test_strict_checkpoint_initialization_copies_every_tensor(tmp_path):
    checkpoint_path = tmp_path / "source.ckpt"
    duo = _checkpoint(checkpoint_path)
    module = _module(checkpoint_path)

    report = module.initialize_from_cfm_checkpoint()

    assert report.global_step == 345
    assert report.epoch == 12
    assert report.source_sd_type == "lag"
    assert report.loaded_tensors == len(duo.state_dict())
    assert report.source_datamodule_hparams == {
        "train_val_test_split": [100, 20, 20],
        "k": 8,
        "batch_size": 4,
    }
    for key, expected in duo.state_dict().items():
        assert torch.equal(module.net.state_dict()[key], expected), key

    report_path = write_initialization_report(report, tmp_path / "run")
    assert '"train_val_test_split"' in report_path.read_text()


def test_source_loss_mismatch_fails_before_training(tmp_path):
    checkpoint_path = tmp_path / "source.ckpt"
    _checkpoint(checkpoint_path, sd_type="ecld")
    module = _module(checkpoint_path)

    with pytest.raises(ValueError, match="source loss mismatch"):
        module.initialize_from_cfm_checkpoint()
