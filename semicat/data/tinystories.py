"""TinyStories DataModule.

Reconstructed to match the interface recorded in the TinyStories A100 run's
resolved Hydra config (``semicat.data.tinystories.TinyStoriesDataModule`` with
``cache_dir / batch_size / max_length / tokenizer_name / dataset_name /
num_workers / num_proc / pin_memory``). It mirrors the LM1B/OWT pipeline in
``lm1b.py``: tokenize each story with a GPT-2 tokenizer, append EOS, concatenate,
and regroup into ``max_length`` windows wrapped in ``[BOS] ... [EOS]``. Unlike
LM1B there is no text de-tokenization step (TinyStories ships clean text), and
the dataset has explicit ``train`` / ``validation`` splits.
"""

import itertools
import os

from lightning import LightningDataModule
import datasets
import tokenizers
import transformers
import torch
from torch.utils.data import DataLoader


class TinyStoriesDataModule(LightningDataModule):
    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 32,
        max_length: int = 256,
        tokenizer_name: str = "gpt2",
        dataset_name: str = "roneneldan/TinyStories",
        num_workers: int = 4,
        num_proc: int | None = None,
        pin_memory: bool = True,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ------------------------------------------------------------------ tokenizer
    def _load_tokenizer(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name, trust_remote_code=True
        )
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        if isinstance(
            self.tokenizer,
            (transformers.GPT2TokenizerFast, transformers.GPT2Tokenizer),
        ):
            self.tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
                (self.tokenizer.bos_token, self.tokenizer.bos_token_id),
                (self.tokenizer.eos_token, self.tokenizer.eos_token_id),
            )
        if self.tokenizer.bos_token is None:
            if self.tokenizer.cls_token is None:
                raise AttributeError(f"Tokenizer must have a bos_token or cls_token: {self.tokenizer}")
            self.tokenizer.bos_token = self.tokenizer.cls_token
        if self.tokenizer.eos_token is None:
            if self.tokenizer.sep_token is None:
                raise AttributeError(f"Tokenizer must have an eos_token or sep_token: {self.tokenizer}")
            self.tokenizer.eos_token = self.tokenizer.sep_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # -------------------------------------------------------------------- dataset
    def _load_dataset(self, split: str):
        assert self.tokenizer is not None, "need tokenizer"
        cache_dir = self.hparams.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        end_file = os.path.join(
            cache_dir, f"tinystories_{self.hparams.max_length}_{split}_processed"
        )
        if os.path.exists(end_file):
            print(f"Loading processed dataset from {end_file}")
            return datasets.load_from_disk(end_file)

        hf_split = split
        if self.hparams.max_samples is not None:
            hf_split = f"{split}[:{self.hparams.max_samples}]"
        dataset = datasets.load_dataset(
            self.hparams.dataset_name,
            streaming=False,
            split=hf_split,
            keep_in_memory=False,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token)[0]
        num_proc = self.hparams.num_proc or 1

        def preprocess(example):
            tokens = self.tokenizer(
                example["text"],
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )
            return {"input_ids": [t + [self.eos] for t in tokens["input_ids"]]}

        dataset = dataset.map(
            preprocess, batched=True, num_proc=num_proc,
            load_from_cache_file=True, desc="Tokenizing", batch_size=1024,
        )
        dataset = dataset.remove_columns(
            [c for c in dataset.column_names if c != "input_ids"]
        )

        def group_texts(examples):
            concatenated = list(itertools.chain(*examples["input_ids"]))
            new_block_size = self.hparams.max_length - 2  # [BOS] and [EOS]
            total_length = (len(concatenated) // new_block_size) * new_block_size
            values = [
                [self.bos] + concatenated[i : i + new_block_size] + [self.eos]
                for i in range(0, total_length, new_block_size)
            ]
            return {"input_ids": values}

        dataset = dataset.map(
            group_texts, batched=True, num_proc=num_proc,
            load_from_cache_file=True, desc="Grouping texts",
        )
        print(f"Saving processed dataset to {end_file}")
        dataset.save_to_disk(end_file)
        return dataset

    def setup(self, stage: str | None = None):
        self._load_tokenizer()
        self.train_dataset = self._load_dataset("train")
        self.val_dataset = self._load_dataset("validation")
        self.test_dataset = self.val_dataset

    # --------------------------------------------------------------------- detok
    def tensor_to_strings(self, batch: torch.Tensor) -> list[str]:
        assert self.tokenizer is not None, "need tokenizer"
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    # ---------------------------------------------------------------- dataloaders
    @staticmethod
    def _collate(examples) -> dict[str, torch.Tensor]:
        # Datasets are kept in native (list) format to avoid a numpy>=2.0 /
        # datasets torch-formatter incompatibility; stack to a LongTensor here.
        return {
            "input_ids": torch.tensor(
                [ex["input_ids"] for ex in examples], dtype=torch.long
            )
        }

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self._collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
