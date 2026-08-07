"""Prepare TinyStories into the same binary format used by the Text8 pipeline.

Stage 2 of the baseline plan: reuse the exact model/trainer/evaluator with one
frozen tokenizer shared across AR, MDLM, and BD3-LM.  The tokenizer here is
byte-level (UTF-8 bytes, 256 symbols).  It is frozen by construction -- no vocab
is fit from data -- so the mapping is fully reproducible and language-agnostic.
The absorbing mask / AR BOS id is 256, i.e. the first id above the byte range.

Binary layout matches ``prepare_text8``: ``train.bin``, ``val.bin``,
``test.bin`` (uint16), ``meta.pkl`` (``vocab_size``/``stoi``/``itos``), and
``dataset_metadata.json`` with source hashes and special-token ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np

from bcfm_baselines.data.text8 import sha256


BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{BASE_URL}/TinyStories-train.txt"
VALID_URL = f"{BASE_URL}/TinyStories-valid.txt"
VOCAB_SIZE = 256  # one id per UTF-8 byte; the mask/BOS id is VOCAB_SIZE (256)

_SMOKE_STORY = (
    "Once upon a time there was a little cat. The cat liked to play in the sun. "
    "One day the cat found a red ball and was very happy. The end.\n"
)


def _load_bytes(source: Path | None, url: str, output_dir: Path, cache_name: str) -> bytes:
    if source is not None:
        return Path(source).read_bytes()
    local = output_dir / cache_name
    if not local.exists():
        print(f"downloading {url} -> {local}")
        urllib.request.urlretrieve(url, local)
    return local.read_bytes()


def _encode(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)


def prepare(
    output_dir: Path,
    source_train: Path | None = None,
    source_valid: Path | None = None,
    smoke: bool = False,
    force: bool = False,
    max_train_bytes: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("train.bin", "val.bin", "test.bin", "meta.pkl")]
    if all(path.exists() for path in outputs) and not force:
        print(f"TinyStories is already prepared in {output_dir}; use --force to replace it")
        return

    if smoke:
        train_bytes = (_SMOKE_STORY * 200).encode("utf-8")
        valid_bytes = (_SMOKE_STORY * 40).encode("utf-8")
        source_description = "deterministic synthetic smoke corpus"
        train_hash = valid_hash = None
    else:
        train_bytes = _load_bytes(source_train, TRAIN_URL, output_dir, "TinyStories-train.txt")
        valid_bytes = _load_bytes(source_valid, VALID_URL, output_dir, "TinyStories-valid.txt")
        source_description = {
            "train": str(source_train.resolve()) if source_train else TRAIN_URL,
            "valid": str(source_valid.resolve()) if source_valid else VALID_URL,
        }
        train_hash = sha256(source_train) if source_train else None
        valid_hash = sha256(source_valid) if source_valid else None

    if max_train_bytes is not None:
        train_bytes = train_bytes[:max_train_bytes]

    train = _encode(train_bytes)
    valid = _encode(valid_bytes)
    # Hold-out test comes from the second half of the official validation file so
    # the training text is never seen at test time.
    half = len(valid) // 2
    splits = {
        "train.bin": train,
        "val.bin": valid[:half],
        "test.bin": valid[half:],
    }
    for name, values in splits.items():
        if len(values) == 0:
            raise ValueError(f"{name} is empty; check the source files")
        values.tofile(output_dir / name)

    stoi = {chr(index): index for index in range(VOCAB_SIZE)}
    itos = {index: chr(index) for index in range(VOCAB_SIZE)}
    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": VOCAB_SIZE, "stoi": stoi, "itos": itos}, handle)

    metadata = {
        "dataset": "tinystories",
        "dataset_version": "roneneldan/TinyStories" if not smoke else "synthetic-smoke-v1",
        "preprocessing_version": 1,
        "source": source_description,
        "source_sha256": {"train": train_hash, "valid": valid_hash},
        "tokenizer": "byte",
        "normalization": "none; raw UTF-8 bytes",
        "vocabulary": "utf-8 bytes 0..255",
        "vocabulary_size": VOCAB_SIZE,
        "special_token_ids": {
            "mask": VOCAB_SIZE,
            "ar_bos": VOCAB_SIZE,
            "padding": None,
            "bos_in_dataset": None,
            "eos": None,
        },
        "split_byte_counts": {name.removesuffix(".bin"): int(len(values)) for name, values in splits.items()},
        "binary_dtype": "uint16",
    }
    for filename in (*splits, "meta.pkl"):
        metadata.setdefault("file_sha256", {})[filename] = sha256(output_dir / filename)
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--source-train", type=Path, help="local TinyStories-train.txt")
    parser.add_argument("--source-valid", type=Path, help="local TinyStories-valid.txt")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-train-bytes",
        type=int,
        default=None,
        help="optional cap on training bytes for a smaller run",
    )
    args = parser.parse_args()
    prepare(
        args.output_dir,
        source_train=args.source_train,
        source_valid=args.source_valid,
        smoke=args.smoke,
        force=args.force,
        max_train_bytes=args.max_train_bytes,
    )


if __name__ == "__main__":
    main()
