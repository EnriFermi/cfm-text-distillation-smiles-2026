from __future__ import annotations

from pathlib import Path

import pytest

from bcfm_baselines.data.prepare_text8 import prepare
from bcfm_baselines.data.prepare_tinystories import load_tokenizer
from bcfm_baselines.data.prepare_tinystories import prepare as prepare_tinystories


@pytest.fixture
def text8_smoke_dir(tmp_path: Path) -> Path:
    path = tmp_path / "text8"
    prepare(path, source_zip=None, smoke=True, force=True)
    return path


@pytest.fixture
def tinystories_smoke_dir(tmp_path: Path) -> Path:
    # The smoke corpus is synthetic, but the GPT-2 BPE still has to be resolved
    # once, which needs tiktoken and (on a cold cache) network access.
    try:
        load_tokenizer()
    except Exception as error:  # pragma: no cover - offline / missing dependency
        pytest.skip(f"GPT-2 tokenizer unavailable: {error}")
    path = tmp_path / "tinystories"
    # block_length must equal the smoke configs' dataset.sequence_length: the
    # corpus is packed into whole blocks and refuses a mismatched window.
    prepare_tinystories(path, smoke=True, force=True, block_length=8)
    return path

