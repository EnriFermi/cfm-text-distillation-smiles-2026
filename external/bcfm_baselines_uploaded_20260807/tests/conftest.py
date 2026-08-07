from __future__ import annotations

from pathlib import Path

import pytest

from bcfm_baselines.data.prepare_text8 import prepare
from bcfm_baselines.data.prepare_tinystories import prepare as prepare_tinystories


@pytest.fixture
def text8_smoke_dir(tmp_path: Path) -> Path:
    path = tmp_path / "text8"
    prepare(path, source_zip=None, smoke=True, force=True)
    return path


@pytest.fixture
def tinystories_smoke_dir(tmp_path: Path) -> Path:
    path = tmp_path / "tinystories"
    prepare_tinystories(path, smoke=True, force=True)
    return path

