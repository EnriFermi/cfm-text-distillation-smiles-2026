"""Prepare canonical Text8 into the reference repository's binary format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import urllib.request
import zipfile

import numpy as np

from bcfm_baselines.data.text8 import sha256


TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
EXPECTED_CHARACTERS = " abcdefghijklmnopqrstuvwxyz"


def prepare(output_dir: Path, source_zip: Path | None, smoke: bool, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("train.bin", "val.bin", "test.bin", "meta.pkl")]
    if all(path.exists() for path in outputs) and not force:
        print(f"Text8 is already prepared in {output_dir}; use --force to replace it")
        return
    if smoke:
        text = (EXPECTED_CHARACTERS * 500)[:10_000]
        source_description = "deterministic synthetic smoke corpus"
        source_hash = None
    else:
        if source_zip is None:
            source_zip = output_dir / "text8.zip"
            if not source_zip.exists():
                urllib.request.urlretrieve(TEXT8_URL, source_zip)
        with zipfile.ZipFile(source_zip) as archive:
            text = archive.read("text8").decode("ascii")
        source_description = str(source_zip.resolve())
        source_hash = sha256(source_zip)
        if len(text) != 100_000_000:
            raise ValueError(f"expected 100,000,000 Text8 characters, found {len(text):,}")
    characters = sorted(set(text))
    if characters != list(EXPECTED_CHARACTERS):
        raise ValueError(f"unexpected Text8 vocabulary: {characters}")
    stoi = {character: index for index, character in enumerate(characters)}
    itos = {index: character for character, index in stoi.items()}
    encoded = np.fromiter((stoi[character] for character in text), dtype=np.uint16, count=len(text))
    if smoke:
        train_end, validation_end = 6_000, 8_000
    else:
        train_end, validation_end = 90_000_000, 95_000_000
    splits = {
        "train.bin": encoded[:train_end],
        "val.bin": encoded[train_end:validation_end],
        "test.bin": encoded[validation_end:],
    }
    for filename, values in splits.items():
        values.tofile(output_dir / filename)
    # Keep the exact keys consumed by the original Text8 pipeline.
    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": len(characters), "stoi": stoi, "itos": itos}, handle)
    metadata = {
        "dataset": "text8",
        "dataset_version": "canonical-100m" if not smoke else "synthetic-smoke-v1",
        "preprocessing_version": 2,
        "source": source_description,
        "source_sha256": source_hash,
        "normalization": "none; canonical Text8 is already lowercase a-z plus space",
        "vocabulary": characters,
        "vocabulary_size": len(characters),
        "special_token_ids": {
            "mask": len(characters),
            "ar_bos": len(characters),
            "padding": None,
            "bos_in_dataset": None,
            "eos": None,
        },
        "split_character_counts": {name.removesuffix(".bin"): len(values) for name, values in splits.items()},
        "binary_dtype": "uint16",
    }
    for filename in (*splits, "meta.pkl"):
        metadata.setdefault("file_sha256", {})[filename] = sha256(output_dir / filename)
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/text8"))
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.output_dir, args.source_zip, args.smoke, args.force)


if __name__ == "__main__":
    main()
