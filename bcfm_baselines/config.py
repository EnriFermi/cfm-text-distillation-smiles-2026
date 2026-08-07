"""Small YAML configuration layer with explicit inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    current = OmegaConf.load(path)
    parent_name = current.pop("inherits", None)
    if parent_name is None:
        merged = current
    else:
        parent = OmegaConf.create(load_config(path.parent / str(parent_name)))
        merged = OmegaConf.merge(parent, current)
    return dict(OmegaConf.to_container(merged, resolve=True))

