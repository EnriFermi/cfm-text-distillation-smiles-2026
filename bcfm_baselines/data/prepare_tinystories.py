"""Prepare TinyStories into the same binary format used by the Text8 pipeline.

Stage 2 of the baseline plan: reuse the exact model/trainer/evaluator with one
frozen tokenizer shared across AR, MDLM, and BD3-LM.  The tokenizer here is the
GPT-2 byte-level BPE (50,257 ids) loaded from ``tiktoken``.  It is frozen by
construction -- no vocab is fit from data -- so the mapping is fully
reproducible. ``<|endoftext|>`` (50256) separates stories and serves as the
tokenizer BOS/EOS inside packed sequences; the absorbing mask / model-only AR
context BOS is 50257, i.e. the first id above the tokenizer's range.

The id -> bytes table is copied into ``meta.pkl``, so training, evaluation, and
detokenization never need ``tiktoken`` or network access again; only this script
does, and only the first time (the BPE files land in ``TIKTOKEN_CACHE_DIR``).

Binary layout matches ``prepare_text8``: ``train.bin``, ``val.bin``,
``test.bin`` (uint16 -- every GPT-2 id fits), ``meta.pkl``
(``vocab_size``/``stoi``/``itos``/``token_bytes``), and
``dataset_metadata.json`` with source hashes and special-token ids.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np

from bcfm_baselines.data.text8 import sha256


BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{BASE_URL}/TinyStories-train.txt"
VALID_URL = f"{BASE_URL}/TinyStories-valid.txt"

TOKENIZER = "gpt2"
VOCAB_SIZE = 50257  # GPT-2 BPE; mask/model-only AR context BOS is id 50257
END_OF_TEXT_ID = 50256  # "<|endoftext|>", the story separator used by the source files
END_OF_TEXT_TEXT = "<|endoftext|>"
DOCUMENT_BATCH = 1024  # stories encoded (and flushed to disk) per tokenizer call
BLOCK_LENGTH = 256  # tokens per packed block; must equal dataset.sequence_length
# GPT-2 has no dedicated BOS: transformers reports bos_token_id == eos_token_id ==
# 50256, and the CFM reference wraps every block with it on both sides.
BOS_ID = END_OF_TEXT_ID

_SMOKE_STORY = (
    "Once upon a time there was a little cat. The cat liked to play in the sun. "
    "One day the cat found a red ball and was very happy. The end.\n"
)


EncodeBatch = Callable[[Sequence[str]], list[list[int]]]


def load_tokenizer() -> tuple[EncodeBatch, list[bytes]]:
    """Return the frozen GPT-2 batch encoder and its id -> bytes table."""
    try:
        import tiktoken
    except ImportError as error:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "the GPT-2 tokenizer needs tiktoken: pip install -r requirements-a100.txt"
        ) from error
    encoding = tiktoken.get_encoding(TOKENIZER)
    if encoding.n_vocab != VOCAB_SIZE:
        raise RuntimeError(f"expected a {VOCAB_SIZE}-token GPT-2 vocabulary, got {encoding.n_vocab}")
    token_bytes = [encoding.decode_single_token_bytes(index) for index in range(VOCAB_SIZE)]
    return encoding.encode_ordinary_batch, token_bytes


def _load_bytes(source: Path | None, url: str, output_dir: Path, cache_name: str) -> bytes:
    if source is not None:
        return Path(source).read_bytes()
    local = output_dir / cache_name
    if not local.exists():
        print(f"downloading {url} -> {local}")
        urllib.request.urlretrieve(url, local)
    return local.read_bytes()


def _documents(raw: bytes) -> list[str]:
    """Split a source file on ``<|endoftext|>``, dropping empty pieces."""
    return [
        document
        for document in raw.decode("utf-8", errors="replace").split(END_OF_TEXT_TEXT)
        if document.strip()
    ]


def _encode(
    documents: Sequence[str], encode_batch: EncodeBatch, block_length: int
) -> Iterator[np.ndarray]:
    """Yield whole packed blocks of ``[BOS, content..., EOS]``.

    Matches the CFM reference pipeline (``semicat/data/tinystories.py``): stories
    are tokenized without special tokens, one ``<|endoftext|>`` is appended per
    story, the resulting stream is cut into ``block_length - 2`` token pieces, and
    each piece is wrapped in BOS/EOS.  Blocks are therefore non-overlapping and
    every training sequence carries the same document-boundary structure -- unlike
    the stride-1 windows over a flat stream this used to emit, where a sequence
    began at an arbitrary offset and no BOS was ever seen.

    The leftover tail shorter than one content block is dropped.  The reference
    drops such a tail once per 1,000-story map batch; carrying it across batches
    here keeps a few hundred thousand more tokens and changes nothing else.
    """
    content_length = block_length - 2
    carry: list[int] = []
    for start in range(0, len(documents), DOCUMENT_BATCH):
        for piece in encode_batch(documents[start : start + DOCUMENT_BATCH]):
            carry.extend(piece)
            carry.append(END_OF_TEXT_ID)
        count = len(carry) // content_length
        if count == 0:
            continue
        content = np.asarray(carry[: count * content_length], dtype=np.uint16)
        del carry[: count * content_length]
        blocks = np.empty((count, block_length), dtype=np.uint16)
        blocks[:, 0] = BOS_ID
        blocks[:, 1:-1] = content.reshape(count, content_length)
        blocks[:, -1] = END_OF_TEXT_ID
        yield blocks.reshape(-1)


def _write(
    path: Path, documents: Sequence[str], encode_batch: EncodeBatch, block_length: int
) -> int:
    """Stream-encode ``documents`` into ``path``; the train file is ~1.9 GB of text."""
    total = 0
    with path.open("wb") as handle:
        for chunk in _encode(documents, encode_batch, block_length):
            chunk.tofile(handle)
            total += int(chunk.size)
    if total == 0:
        raise ValueError(
            f"{path.name} is empty: the source holds fewer than {block_length - 2} tokens"
        )
    return total


def _already_prepared(
    output_dir: Path,
    outputs: Sequence[Path],
    block_length: int,
    preparation: dict,
) -> bool:
    """True only when every output exists *and* matches this tokenizer and packing."""
    if not all(path.exists() for path in outputs):
        return False
    metadata_path = output_dir / "dataset_metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    return (
        metadata.get("tokenizer") == TOKENIZER
        and metadata.get("packed_block_length") == block_length
        and metadata.get("preparation") == preparation
    )


def prepare(
    output_dir: Path,
    source_train: Path | None = None,
    source_valid: Path | None = None,
    smoke: bool = False,
    force: bool = False,
    max_train_bytes: int | None = None,
    block_length: int = BLOCK_LENGTH,
) -> None:
    if block_length < 3:
        raise ValueError("block_length must leave room for BOS, one token, and EOS")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("train.bin", "val.bin", "test.bin", "meta.pkl")]
    if smoke:
        requested_source: str | dict[str, str] = "deterministic synthetic smoke corpus"
        requested_hashes = {"train": None, "valid": None}
    else:
        requested_source = {
            "train": str(source_train.resolve()) if source_train else TRAIN_URL,
            "valid": str(source_valid.resolve()) if source_valid else VALID_URL,
        }
        requested_hashes = {
            "train": sha256(source_train) if source_train else None,
            "valid": sha256(source_valid) if source_valid else None,
        }
    preparation = {
        "smoke": smoke,
        "source": requested_source,
        "source_sha256": requested_hashes,
        "max_train_bytes": max_train_bytes,
        "block_length": block_length,
    }
    if _already_prepared(output_dir, outputs, block_length, preparation) and not force:
        print(f"TinyStories is already prepared in {output_dir}; use --force to replace it")
        return
    if all(path.exists() for path in outputs):
        print(
            f"re-encoding {output_dir}: the existing bins are not {TOKENIZER}-tokenized "
            f"into {block_length}-token packed blocks"
        )

    if smoke:
        # Separated like the real files so the smoke path exercises the same
        # document splitting rather than a degenerate single-story corpus.
        story = (_SMOKE_STORY + END_OF_TEXT_TEXT).encode("utf-8")
        train_bytes = story * 200
        valid_bytes = story * 40
        source_description = requested_source
        train_hash = valid_hash = None
    else:
        train_bytes = _load_bytes(source_train, TRAIN_URL, output_dir, "TinyStories-train.txt")
        valid_bytes = _load_bytes(source_valid, VALID_URL, output_dir, "TinyStories-valid.txt")
        source_description = requested_source
        train_hash = requested_hashes["train"]
        valid_hash = requested_hashes["valid"]

    if max_train_bytes is not None:
        train_bytes = train_bytes[:max_train_bytes]

    encode_batch, token_bytes = load_tokenizer()
    train_documents = _documents(train_bytes)
    valid_documents = _documents(valid_bytes)
    del train_bytes, valid_bytes
    if len(valid_documents) < 2:
        raise ValueError(
            "the validation source must hold at least two <|endoftext|>-separated "
            f"stories to split into val and test, found {len(valid_documents)}"
        )
    # Hold-out test comes from the second half of the official validation file so
    # the training text is never seen at test time.  Splitting on story rather
    # than token boundaries keeps every story wholly inside one split.
    half = len(valid_documents) // 2
    splits = {
        "train.bin": train_documents,
        "val.bin": valid_documents[:half],
        "test.bin": valid_documents[half:],
    }
    token_counts = {
        name: _write(output_dir / name, documents, encode_batch, block_length)
        for name, documents in splits.items()
    }

    itos = {index: value.decode("utf-8", errors="replace") for index, value in enumerate(token_bytes)}
    # ``stoi`` exists for structural parity with the Text8 meta; detokenization
    # goes through ``token_bytes`` because a BPE piece is not always valid UTF-8.
    stoi = {text: index for index, text in itos.items()}
    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(
            {
                "vocab_size": VOCAB_SIZE,
                "stoi": stoi,
                "itos": itos,
                "token_bytes": token_bytes,
            },
            handle,
        )

    metadata = {
        "dataset": "tinystories",
        "dataset_version": "roneneldan/TinyStories" if not smoke else "synthetic-smoke-v1",
        "preprocessing_version": 3,
        "source": source_description,
        "source_sha256": {"train": train_hash, "valid": valid_hash},
        "preparation": preparation,
        "tokenizer": TOKENIZER,
        "tokenizer_source": f"tiktoken:{TOKENIZER}",
        "normalization": "none; UTF-8 text split on <|endoftext|>",
        "vocabulary": "gpt-2 byte-level bpe, ids 0..50256",
        "vocabulary_size": VOCAB_SIZE,
        # Non-overlapping [BOS, block_length - 2 content tokens, EOS] blocks, as in
        # the CFM reference. Readers must consume the bins block-aligned.
        "packed_block_length": block_length,
        "packed_content_length": block_length - 2,
        "sequence_layout": "packed_blocks",
        "special_token_ids": {
            "mask": VOCAB_SIZE,
            "ar_bos": VOCAB_SIZE,
            "padding": None,
            "bos_in_dataset": BOS_ID,
            "eos": END_OF_TEXT_ID,
        },
        "split_token_counts": {
            name.removesuffix(".bin"): count for name, count in token_counts.items()
        },
        "split_block_counts": {
            name.removesuffix(".bin"): count // block_length
            for name, count in token_counts.items()
        },
        "split_document_counts": {
            name.removesuffix(".bin"): len(documents) for name, documents in splits.items()
        },
        # Per-token NLL is not per-character NLL under BPE; these counts are what
        # converts a reported bits/token into bits/character after the fact.
        "split_character_counts": {
            name.removesuffix(".bin"): sum(len(document) for document in documents)
            for name, documents in splits.items()
        },
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
    parser.add_argument(
        "--block-length",
        type=int,
        default=BLOCK_LENGTH,
        help="tokens per packed block; must equal the config's dataset.sequence_length",
    )
    args = parser.parse_args()
    prepare(
        args.output_dir,
        source_train=args.source_train,
        source_valid=args.source_valid,
        smoke=args.smoke,
        force=args.force,
        max_train_bytes=args.max_train_bytes,
        block_length=args.block_length,
    )


if __name__ == "__main__":
    main()
