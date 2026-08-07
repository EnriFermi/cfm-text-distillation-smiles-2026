"""Byte-level TinyStories corpus sharing the Text8 binary/window machinery."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

from bcfm_baselines.data.text8 import Text8Corpus

VOCAB_SIZE = 256


class TinyStoriesCorpus(Text8Corpus):
    """TinyStories with a frozen UTF-8 byte tokenizer.

    Reuses ``Text8Corpus`` for the memory-mapped ``uint16`` layout, reproducible
    random windows, and sequential batching.  Only decoding and reported
    metadata differ: ids are raw bytes, so decoding round-trips through UTF-8
    rather than a single-character table.  ``reference_split`` defaults to
    ``False`` because TinyStories ships explicit train/val/test bins.
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 256,
        reference_split: bool = False,
        split_lengths: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(data_dir, sequence_length, reference_split, split_lengths)

    def decode(self, tokens: Tensor) -> list[str]:
        return [
            bytes(token & 0xFF for token in row if token < VOCAB_SIZE).decode(
                "utf-8", errors="replace"
            )
            for row in tokens.cpu().tolist()
        ]

    def metadata(self) -> dict:
        base = super().metadata()
        base["tokenizer"] = "byte"
        base["vocabulary"] = "utf-8 bytes 0..255"
        return base
