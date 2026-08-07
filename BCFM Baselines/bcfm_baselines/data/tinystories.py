"""GPT-2 tokenized TinyStories corpus, read as non-overlapping packed blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from bcfm_baselines.data.text8 import Split, Text8Corpus

VOCAB_SIZE = 50257
END_OF_TEXT_ID = 50256
BOS_ID = END_OF_TEXT_ID


class TinyStoriesCorpus(Text8Corpus):
    """TinyStories under the frozen GPT-2 byte-level BPE.

    Reuses ``Text8Corpus`` for the memory-mapped ``uint16`` layout and split
    handling, but not its windowing.  ``prepare_tinystories`` writes the corpus
    as a sequence of ``[BOS, 254 content tokens, EOS]`` blocks, so windows here
    are block-aligned rather than free-floating: every training and evaluation
    sequence is one whole block, exactly as in the CFM reference.  Sampling an
    arbitrary offset instead would put BOS/EOS at a random position and teach the
    model a document structure the reference never sees.

    Detokenization concatenates the id -> bytes table frozen into ``meta.pkl`` at
    preparation time and decodes UTF-8 once, dropping the special ids the way
    ``skip_special_tokens=True`` does on the reference side.  ``reference_split``
    defaults to ``False`` because TinyStories ships explicit train/val/test bins.
    """

    tokenizer_name = "gpt2"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 256,
        reference_split: bool = False,
        split_lengths: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(data_dir, sequence_length, reference_split, split_lengths)
        token_bytes = self.meta.get("token_bytes")
        if token_bytes is None or len(token_bytes) != self.vocab_size:
            raise ValueError(
                f"{self.data_dir}/meta.pkl carries no GPT-2 id->bytes table, so it "
                "predates the GPT-2 tokenizer; re-run "
                "`python -m bcfm_baselines.data.prepare_tinystories --force`"
            )
        self.token_bytes: list[bytes] = token_bytes
        self.block_length = int(
            self._dataset_metadata().get("packed_block_length", 0)
        )
        if self.block_length != sequence_length:
            raise ValueError(
                f"{self.data_dir} holds {self.block_length or 'unpacked'} token blocks "
                f"but sequence_length is {sequence_length}; re-run "
                f"`python -m bcfm_baselines.data.prepare_tinystories --force "
                f"--block-length {sequence_length}`"
            )
        for name, array in self.arrays.items():
            if len(array) < self.block_length:
                raise ValueError(f"{name} split holds no complete block")

    def _dataset_metadata(self) -> dict:
        path = self.data_dir / "dataset_metadata.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _blocks(self, split: Split) -> int:
        return len(self.arrays[split]) // self.block_length

    def _gather(self, split: Split, indices) -> Tensor:
        data = self.arrays[split]
        length = self.block_length
        batch = np.stack(
            [
                np.asarray(data[index * length : (index + 1) * length], dtype=np.int64)
                for index in indices
            ]
        )
        return torch.from_numpy(batch)

    def random_batch(
        self,
        split: Split,
        batch_size: int,
        generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        indices = torch.randint(0, self._blocks(split), (batch_size,), generator=generator)
        batch = self._gather(split, indices.tolist())
        return batch.to(device=device, non_blocking=True)

    def sequential_batches(
        self,
        split: Split,
        batch_size: int,
        max_batches: int | None = None,
        stride: int | None = None,
    ):
        # `stride` is accepted for interface parity with Text8Corpus; blocks are
        # already disjoint, so anything other than one block per step would either
        # skip or re-score whole documents.
        del stride
        total = self._blocks(split)
        for batch_number, first in enumerate(range(0, total, batch_size)):
            if max_batches is not None and batch_number >= max_batches:
                break
            yield self._gather(split, range(first, min(first + batch_size, total)))

    def decode(self, tokens: Tensor) -> list[str]:
        special = {END_OF_TEXT_ID}
        return [
            b"".join(
                self.token_bytes[int(token)]
                for token in row
                if token < self.vocab_size and int(token) not in special
            )
            .decode("utf-8", errors="replace")
            for row in tokens.cpu().tolist()
        ]

    def _vocabulary_description(self) -> str:
        # Listing 50,257 pieces in every run's dataset_metadata.json is noise.
        return "gpt-2 byte-level bpe, ids 0..50256"

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "window_stride": self.block_length,
            "sequence_layout": "packed_blocks",
            "blocks_in_use": {name: self._blocks(name) for name in self.arrays},
        }
