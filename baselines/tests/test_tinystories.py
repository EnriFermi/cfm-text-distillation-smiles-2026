"""Stage 2 TinyStories data layer: preparation, corpus, and end-to-end smoke."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import pickle

import pytest
import torch

from bcfm_baselines.checkpoint import load_checkpoint_model
from bcfm_baselines.config import load_config
from bcfm_baselines.data import TinyStoriesCorpus, create_corpus
from bcfm_baselines.data.prepare_tinystories import VOCAB_SIZE
from bcfm_baselines.evaluate import evaluate
from bcfm_baselines.factory import create_model
from bcfm_baselines.models.generative_text import GenerationConfig
from bcfm_baselines.train import train


SMOKE_CONFIGS = [
    "configs/smoke/tinystories_ar.yaml",
    "configs/smoke/tinystories_mdlm.yaml",
    "configs/smoke/tinystories_bd3lm.yaml",
]


def test_prepare_writes_byte_vocabulary(tinystories_smoke_dir: Path) -> None:
    with (tinystories_smoke_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    assert meta["vocab_size"] == VOCAB_SIZE
    for name in ("train.bin", "val.bin", "test.bin", "dataset_metadata.json"):
        assert (tinystories_smoke_dir / name).exists()


def test_corpus_round_trips_utf8(tinystories_smoke_dir: Path) -> None:
    corpus = TinyStoriesCorpus(tinystories_smoke_dir, sequence_length=8)
    assert corpus.vocab_size == VOCAB_SIZE
    assert corpus.metadata()["tokenizer"] == "byte"
    generator = torch.Generator().manual_seed(0)
    batch = corpus.random_batch("train", 2, generator)
    assert batch.shape == (2, 8)
    assert int(batch.max()) < VOCAB_SIZE
    texts = corpus.decode(batch)
    assert len(texts) == 2 and all(isinstance(text, str) for text in texts)


def test_create_corpus_selects_tinystories(tinystories_smoke_dir: Path) -> None:
    config = load_config("configs/smoke/tinystories_mdlm.yaml")
    config["dataset"]["data_dir"] = str(tinystories_smoke_dir)
    corpus = create_corpus(config)
    assert isinstance(corpus, TinyStoriesCorpus)


@pytest.mark.parametrize("config_path", SMOKE_CONFIGS)
def test_one_step_train_eval_on_tinystories(
    tinystories_smoke_dir: Path, tmp_path: Path, config_path: str
) -> None:
    config = load_config(config_path)
    config["dataset"]["data_dir"] = str(tinystories_smoke_dir)
    run_dir = tmp_path / Path(config_path).stem
    train(config, run_dir, resume="none")
    checkpoint = run_dir / "checkpoints" / "last.pt"
    assert checkpoint.exists()
    # Both last and best checkpoints are produced, and TensorBoard logs exist.
    best = run_dir / "checkpoints" / "best.pt"
    assert best.exists()
    assert load_checkpoint_model(best)[1]["monitored_metric"] in {"nll", "nelbo"}
    assert (run_dir / "tensorboard").exists()
    assert any((run_dir / "tensorboard").iterdir())

    model, payload = load_checkpoint_model(checkpoint)
    assert payload["step"] == 1
    generation = GenerationConfig(**config["sampling"])
    tokens = model.generate(generation).tokens
    assert int(tokens.max()) < VOCAB_SIZE

    results = evaluate(
        Namespace(
            checkpoint=checkpoint,
            device="cpu",
            eval_batch_size=2,
            eval_batches=1,
            num_samples=2,
            seed=7,
            num_steps=2,
            steps_per_block=2 if config["model"]["type"] == "bd3lm" else None,
        )
    )
    assert results["model_type"] == config["model"]["type"]
    assert results["dataset_metadata"]["tokenizer"] == "byte"
    assert all(isinstance(text, str) for text in results["sample_texts"])


def test_backbone_uses_byte_vocabulary() -> None:
    model = create_model(load_config("configs/tinystories/ar.yaml"))
    assert model.backbone.config.vocab_size == VOCAB_SIZE + 1
    assert model.data_vocab_size == VOCAB_SIZE
    assert model.mask_token_id == VOCAB_SIZE
