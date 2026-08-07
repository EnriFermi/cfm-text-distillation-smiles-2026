"""Stage 2 TinyStories data layer: preparation, corpus, and end-to-end smoke."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import pickle

import pytest
import torch

from bcfm_baselines.checkpoint import load_checkpoint_model
from bcfm_baselines.config import load_config
from bcfm_baselines.data import TinyStoriesCorpus, create_corpus
from bcfm_baselines.data.prepare_tinystories import END_OF_TEXT_ID, VOCAB_SIZE
from bcfm_baselines.evaluate import evaluate
from bcfm_baselines.factory import create_model
from bcfm_baselines.models.generative_text import GenerationConfig
from bcfm_baselines.train import train


SMOKE_CONFIGS = [
    "configs/smoke/tinystories_ar.yaml",
    "configs/smoke/tinystories_mdlm.yaml",
    "configs/smoke/tinystories_bd3lm.yaml",
]


def test_prepare_writes_gpt2_vocabulary(tinystories_smoke_dir: Path) -> None:
    with (tinystories_smoke_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    assert meta["vocab_size"] == VOCAB_SIZE
    # The id -> bytes table is what makes detokenization work without tiktoken.
    assert len(meta["token_bytes"]) == VOCAB_SIZE
    assert meta["token_bytes"][END_OF_TEXT_ID] == b"<|endoftext|>"
    for name in ("train.bin", "val.bin", "test.bin", "dataset_metadata.json"):
        assert (tinystories_smoke_dir / name).exists()
    metadata = json.loads((tinystories_smoke_dir / "dataset_metadata.json").read_text())
    assert metadata["tokenizer"] == "gpt2"
    assert metadata["special_token_ids"] == {
        "mask": VOCAB_SIZE,
        "ar_bos": VOCAB_SIZE,
        "padding": None,
        "bos_in_dataset": END_OF_TEXT_ID,
        "eos": END_OF_TEXT_ID,
    }
    assert metadata["packed_block_length"] == 8
    assert metadata["preparation"] == {
        "smoke": True,
        "source": "deterministic synthetic smoke corpus",
        "source_sha256": {"train": None, "valid": None},
        "max_train_bytes": None,
        "block_length": 8,
    }
    assert all(count % 8 == 0 for count in metadata["split_token_counts"].values())


def test_prepare_packs_bos_eos_wrapped_blocks(tinystories_smoke_dir: Path) -> None:
    """Every block is [BOS, content..., EOS], as in the CFM reference."""
    import numpy as np

    data = np.memmap(tinystories_smoke_dir / "train.bin", dtype=np.uint16, mode="r")
    blocks = np.asarray(data).reshape(-1, 8)
    assert len(blocks) > 1
    assert (blocks[:, 0] == END_OF_TEXT_ID).all()
    assert (blocks[:, -1] == END_OF_TEXT_ID).all()


def test_corpus_rejects_a_mismatched_block_length(tinystories_smoke_dir: Path) -> None:
    with pytest.raises(ValueError, match="block-length"):
        TinyStoriesCorpus(tinystories_smoke_dir, sequence_length=16)


def test_corpus_round_trips_utf8(tinystories_smoke_dir: Path) -> None:
    corpus = TinyStoriesCorpus(tinystories_smoke_dir, sequence_length=8)
    assert corpus.vocab_size == VOCAB_SIZE
    assert corpus.metadata()["tokenizer"] == "gpt2"
    generator = torch.Generator().manual_seed(0)
    batch = corpus.random_batch("train", 2, generator)
    assert batch.shape == (2, 8)
    assert int(batch.max()) < VOCAB_SIZE
    # Blocks are aligned, so BOS/EOS always land on the first and last position.
    assert (batch[:, 0] == END_OF_TEXT_ID).all()
    assert (batch[:, -1] == END_OF_TEXT_ID).all()
    texts = corpus.decode(batch)
    assert len(texts) == 2 and all(isinstance(text, str) for text in texts)
    # decode() strips the special ids the way skip_special_tokens=True does.
    assert not any("<|endoftext|>" in text for text in texts)


def test_corpus_rejects_a_pre_gpt2_meta(tinystories_smoke_dir: Path) -> None:
    with (tinystories_smoke_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    del meta["token_bytes"]
    with (tinystories_smoke_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)
    with pytest.raises(ValueError, match="prepare_tinystories"):
        TinyStoriesCorpus(tinystories_smoke_dir, sequence_length=8)


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
    if config["model"]["type"] in {"mdlm", "bd3lm"}:
        assert model.ignore_bos is True
        assert tokens[:, 0].eq(END_OF_TEXT_ID).all()

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
    assert results["dataset_metadata"]["tokenizer"] == "gpt2"
    assert all(isinstance(text, str) for text in results["sample_texts"])


def test_backbone_uses_gpt2_vocabulary() -> None:
    model = create_model(load_config("configs/tinystories/ar.yaml"))
    assert model.backbone.config.vocab_size == VOCAB_SIZE + 1
    assert model.data_vocab_size == VOCAB_SIZE
    assert model.mask_token_id == VOCAB_SIZE


def test_diffusion_keeps_tinystories_bos_clean_and_excludes_it_at_eval() -> None:
    config = load_config("configs/smoke/tinystories_bd3lm.yaml")
    model = create_model(config)
    clean = torch.randint(0, VOCAB_SIZE, (2, 8))
    clean[:, 0] = END_OF_TEXT_ID
    times = torch.full((2, 2), 0.5)
    corruption = torch.zeros_like(clean, dtype=torch.float)

    training = model.compute_loss(
        clean, time_values=times, corruption_uniform=corruption
    )
    assert training["token_count"].item() == 16
    assert training["masked_fraction"].item() == pytest.approx(14 / 16)

    model.eval()
    validation = model.compute_loss(
        clean, time_values=times, corruption_uniform=corruption
    )
    assert validation["token_count"].item() == 14
    assert validation["masked_fraction"].item() == pytest.approx(1.0)
