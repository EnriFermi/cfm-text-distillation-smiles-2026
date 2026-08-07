"""Deterministic Text8 binary format shared by every baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import torch
from torch import Tensor


Split = Literal["train", "validation", "test"]


class Text8Corpus:
    """Memory-mapped Text8 corpus with reproducible random window sampling.

    ``reference_split`` reproduces the exact split lengths and validation
    slicing found in the reference repository: 351,563 training characters,
    then 20,000 validation and 19,063 test characters from ``val.bin``.
    """

    tokenizer_name = "character"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 256,
        reference_split: bool = True,
        split_lengths: tuple[int, int, int] = (351_563, 20_000, 19_063),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        with (self.data_dir / "meta.pkl").open("rb") as handle:
            self.meta = pickle.load(handle)
        self.stoi: dict[str, int] = self.meta["stoi"]
        self.itos: dict[int, str] = self.meta["itos"]
        self.vocab_size = int(self.meta["vocab_size"])
        train = np.memmap(self.data_dir / "train.bin", dtype=np.uint16, mode="r")
        validation = np.memmap(self.data_dir / "val.bin", dtype=np.uint16, mode="r")
        test_path = self.data_dir / "test.bin"
        test = np.memmap(test_path, dtype=np.uint16, mode="r") if test_path.exists() else None
        if reference_split:
            train_length, validation_length, test_length = split_lengths
            self.arrays = {
                "train": train[:train_length],
                "validation": validation[:validation_length],
                "test": validation[validation_length : validation_length + test_length],
            }
        else:
            self.arrays = {
                "train": train,
                "validation": validation,
                "test": validation if test is None else test,
            }
        for name, array in self.arrays.items():
            if len(array) < sequence_length:
                raise ValueError(f"{name} split is shorter than sequence_length")

    def random_batch(
        self,
        split: Split,
        batch_size: int,
        generator: torch.Generator,
        *,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        data = self.arrays[split]
        starts = torch.randint(
            0,
            len(data) - self.sequence_length + 1,
            (batch_size,),
            generator=generator,
        )
        batch = np.stack(
            [np.asarray(data[int(start) : int(start) + self.sequence_length], dtype=np.int64) for start in starts]
        )
        return torch.from_numpy(batch).to(device=device, non_blocking=True)

    def sequential_batches(
        self,
        split: Split,
        batch_size: int,
        max_batches: int | None = None,
        stride: int | None = None,
    ):
        """Yield deterministic windows, ``stride`` tokens apart.

        The default stride is the full sequence length, so a capped validation
        pass covers ``batch_size * max_batches * sequence_length`` distinct
        tokens.  With the stride-1 windows this used to yield, 1,600 sequences
        overlapped so heavily that they covered barely 1,855 distinct tokens --
        an evaluation of one paragraph, repeated.  Pass ``stride=1`` to recover
        the old behaviour.
        """
        data = self.arrays[split]
        step = self.sequence_length if stride is None else stride
        if step <= 0:
            raise ValueError("stride must be positive")
        starts = range(0, len(data) - self.sequence_length + 1, step)
        batch_number = 0
        for first in range(0, len(starts), batch_size):
            if max_batches is not None and batch_number >= max_batches:
                break
            window_starts = starts[first : first + batch_size]
            batch = np.stack(
                [
                    np.asarray(data[start : start + self.sequence_length], dtype=np.int64)
                    for start in window_starts
                ]
            )
            yield torch.from_numpy(batch)
            batch_number += 1

    def decode(self, tokens: Tensor) -> list[str]:
        return ["".join(self.itos[int(token)] for token in row) for row in tokens.cpu().tolist()]

    def _vocabulary_description(self) -> list[str] | str:
        return [self.itos[index] for index in range(self.vocab_size)]

    def metadata(self) -> dict:
        metadata_path = self.data_dir / "dataset_metadata.json"
        source = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return {
            **source,
            "tokenizer": self.tokenizer_name,
            "vocabulary_size": self.vocab_size,
            "vocabulary": self._vocabulary_description(),
            "sequence_length": self.sequence_length,
            "split_lengths_in_use": {name: len(value) for name, value in self.arrays.items()},
            "window_stride": self.sequence_length,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
