import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from semicat.data.text8 import Text8DataModule, Text8Dataset


def _write_fixture(tmp_path):
    meta = {
        "vocab_size": 27,
        "stoi": {chr(97 + index): index for index in range(27)},
        "itos": {index: chr(97 + index) for index in range(27)},
    }
    with (tmp_path / "meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)
    np.arange(30, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.arange(100, 140, dtype=np.uint16).tofile(tmp_path / "val.bin")


def _cpu_module(tmp_path, **kwargs):
    module = Text8DataModule(data_dir=str(tmp_path), **kwargs)
    module.trainer = SimpleNamespace(
        strategy=SimpleNamespace(root_device=torch.device("cpu")),
        world_size=1,
    )
    module.setup("fit")
    return module


def test_upstream_text8_dataset_uses_stride_one_windows():
    dataset = Text8Dataset(torch.arange(10), k=4, dim=27)

    assert len(dataset) == 7
    assert torch.equal(dataset[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(dataset[1], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(dataset[6], torch.tensor([6, 7, 8, 9]))


def test_upstream_datamodule_split_and_loader_semantics(tmp_path):
    meta = {
        "vocab_size": 27,
        "stoi": {chr(97 + index): index for index in range(27)},
        "itos": {index: chr(97 + index) for index in range(27)},
    }
    with (tmp_path / "meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)
    np.arange(30, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.arange(100, 140, dtype=np.uint16).tofile(tmp_path / "val.bin")

    module = Text8DataModule(
        train_val_test_split=(12, 10, 10),
        k=4,
        data_dir=str(tmp_path),
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        prefetch_factor=2,
    )
    module.trainer = SimpleNamespace(
        strategy=SimpleNamespace(root_device=torch.device("cpu")),
        world_size=1,
    )
    module.setup("fit")

    assert torch.equal(module.data_train.all_data, torch.arange(12))
    assert torch.equal(module.data_val.all_data, torch.arange(100, 110))
    assert torch.equal(module.data_test.all_data, torch.arange(110, 120))
    assert len(module.data_train) == 9
    assert len(module.data_val) == len(module.data_test) == 7
    assert isinstance(module.train_dataloader().sampler, RandomSampler)
    assert isinstance(module.val_dataloader().sampler, SequentialSampler)
    assert isinstance(module.test_dataloader().sampler, SequentialSampler)
    assert module.train_dataloader().num_workers == 0
    assert module.test_dataloader().num_workers == 1


def test_split_units_defaults_to_upstream_char_slicing(tmp_path):
    _write_fixture(tmp_path)

    module = _cpu_module(tmp_path, train_val_test_split=(12, 10, 10), k=4, batch_size=2)

    assert module.hparams.split_units == "chars"
    assert torch.equal(module.data_train.all_data, torch.arange(12))


def test_split_units_sequences_scales_the_split_by_k(tmp_path):
    _write_fixture(tmp_path)

    module = _cpu_module(
        tmp_path,
        train_val_test_split=(2, 2, 2),
        k=4,
        batch_size=2,
        split_units="sequences",
    )

    # 2 sequences of 4 tokens -> 8 characters per split, not 2
    assert torch.equal(module.data_train.all_data, torch.arange(8))
    assert torch.equal(module.data_val.all_data, torch.arange(100, 108))
    assert torch.equal(module.data_test.all_data, torch.arange(108, 116))
    assert len(module.data_train) == 5  # stride-one windows over 8 characters


def test_split_units_sequences_covers_the_whole_corpus(tmp_path):
    """The canonical text8 split is 0.39% of train.bin in the upstream mode."""
    _write_fixture(tmp_path)

    legacy = _cpu_module(tmp_path, train_val_test_split=(3, 2, 2), k=4, batch_size=2)
    fixed = _cpu_module(
        tmp_path,
        train_val_test_split=(3, 2, 2),
        k=4,
        batch_size=2,
        split_units="sequences",
    )

    assert len(legacy.data_train.all_data) == 3
    assert len(fixed.data_train.all_data) == 12


def test_split_units_rejects_unknown_value():
    with pytest.raises(ValueError, match="split_units"):
        Text8DataModule(train_val_test_split=(1, 1, 1), split_units="tokens")
