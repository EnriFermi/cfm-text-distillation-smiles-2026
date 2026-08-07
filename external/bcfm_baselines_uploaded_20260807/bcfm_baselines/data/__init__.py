"""Dataset corpora and a config-driven corpus factory."""

from __future__ import annotations

from typing import Any

from bcfm_baselines.data.text8 import Text8Corpus
from bcfm_baselines.data.tinystories import TinyStoriesCorpus

__all__ = ["Text8Corpus", "TinyStoriesCorpus", "create_corpus"]


def create_corpus(config: dict[str, Any]):
    """Build the corpus selected by ``config['dataset']['name']``.

    Defaults to Text8 so existing checkpoints/configs keep working unchanged.
    """
    dataset = config["dataset"]
    name = str(dataset.get("name", "text8")).lower()
    sequence_length = dataset["sequence_length"]
    if name == "tinystories":
        return TinyStoriesCorpus(
            dataset["data_dir"],
            sequence_length=sequence_length,
            reference_split=dataset.get("reference_split", False),
        )
    if name == "text8":
        return Text8Corpus(
            dataset["data_dir"],
            sequence_length=sequence_length,
            reference_split=dataset.get("reference_split", True),
            split_lengths=tuple(dataset.get("split_lengths", (351_563, 20_000, 19_063))),
        )
    raise ValueError(f"unknown dataset.name {name!r}; expected text8 or tinystories")
