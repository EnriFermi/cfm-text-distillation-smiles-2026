"""Strict CFM-to-BlockDIT checkpoint initialization."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn


@dataclass(frozen=True)
class CFMInitializationReport:
    checkpoint: str
    global_step: int | None
    epoch: int | None
    source_sd_type: str | None
    source_in_shape: list[int] | None
    source_datamodule_hparams: dict[str, Any] | None
    loaded_tensors: int
    loaded_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a config path robustly under Hydra's optional working-dir change."""
    requested = Path(path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(Path.cwd() / requested)
        project_root = os.environ.get("PROJECT_ROOT")
        if project_root:
            candidates.append(Path(project_root) / requested)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"CFM initialization checkpoint not found; tried: {attempted}")


def _net_state_dict(checkpoint_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    net_state: dict[str, torch.Tensor] = {}
    for source_key, value in checkpoint_state.items():
        key = source_key.replace("net._orig_mod.", "net.")
        if not key.startswith("net."):
            continue
        key = key.removeprefix("net.")
        if key in net_state:
            raise ValueError(f"duplicate normalized CFM key: {key}")
        net_state[key] = value
    if not net_state:
        raise ValueError("checkpoint contains no net.* tensors")
    return net_state


def _plain_container(value: Any) -> Any:
    """Convert Lightning/OmegaConf checkpoint metadata to JSON primitives."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {str(key): _plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_container(item) for item in value]
    return value


def initialize_blockdit_from_cfm(
    net: nn.Module,
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_in_shape: tuple[int, ...] | None = None,
    expected_source_sd_type: str | None = None,
) -> CFMInitializationReport:
    """Strict-load only CFM network weights, intentionally not optimizer state."""
    path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected a Lightning checkpoint dict, got {type(checkpoint).__name__}")
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("checkpoint is missing a state_dict")

    hparams = checkpoint.get("hyper_parameters") or {}
    source_sd_type = hparams.get("sd_type")
    source_in_shape = hparams.get("in_shape")
    if expected_source_sd_type is not None and source_sd_type != expected_source_sd_type:
        raise ValueError(
            "source loss mismatch: "
            f"expected sd_type={expected_source_sd_type!r}, checkpoint has {source_sd_type!r}"
        )
    if expected_in_shape is not None and tuple(source_in_shape or ()) != tuple(expected_in_shape):
        raise ValueError(
            "source in_shape mismatch: "
            f"expected {tuple(expected_in_shape)}, checkpoint has {source_in_shape}"
        )

    source_state = _net_state_dict(raw_state)
    target_state = net.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    shape_mismatches = sorted(
        (
            key,
            tuple(source_state[key].shape),
            tuple(target_state[key].shape),
        )
        for key in set(source_state) & set(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "CFM checkpoint is not strictly compatible with BlockDIT: "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}, "
            f"shape_mismatches={shape_mismatches[:12]}"
        )

    net.load_state_dict(source_state, strict=True)
    report = CFMInitializationReport(
        checkpoint=str(path),
        global_step=checkpoint.get("global_step"),
        epoch=checkpoint.get("epoch"),
        source_sd_type=source_sd_type,
        source_in_shape=list(source_in_shape) if source_in_shape is not None else None,
        source_datamodule_hparams=_plain_container(
            checkpoint.get("datamodule_hyper_parameters")
        ),
        loaded_tensors=len(source_state),
        loaded_parameters=sum(parameter.numel() for parameter in net.parameters()),
    )
    del source_state, raw_state, checkpoint
    gc.collect()
    return report


def write_initialization_report(
    report: CFMInitializationReport,
    output_dir: str | os.PathLike[str],
) -> Path:
    path = Path(output_dir) / "cfm_initialization.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path
