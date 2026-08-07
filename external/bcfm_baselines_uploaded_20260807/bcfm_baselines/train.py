"""Non-interactive Text8 trainer designed for Slurm execution and resumption."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

try:  # TensorBoard is a pinned dependency; degrade gracefully if unavailable.
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - only hit when tensorboard is missing
    SummaryWriter = None

from bcfm_baselines.checkpoint import load_model_state
from bcfm_baselines.config import load_config
from bcfm_baselines.data import create_corpus
from bcfm_baselines.factory import create_model


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _device(config: dict[str, Any]) -> torch.device:
    requested = config["training"].get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; smoke configs use device=cpu")
    return torch.device(requested)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _scheduler(optimizer, warmup_steps: int, max_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _clone_model_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _load_pretrained_backbone(model, checkpoint: str | Path, device: torch.device) -> dict[str, Any]:
    """Initialize a BD3-LM backbone from a compatible local MDLM checkpoint."""
    if model.model_type != "bd3lm":
        raise ValueError("training.from_pretrained is only supported for BD3-LM")
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"pretrained MDLM checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    source_type = payload.get("config", {}).get("model", {}).get("type")
    if source_type != "mdlm":
        raise ValueError(
            f"BD3-LM initialization requires an MDLM checkpoint, got {source_type!r}"
        )
    use_ema = payload.get("ema_model") is not None
    source_state = payload["ema_model"] if use_ema else payload["model"]
    backbone_state = {
        name.removeprefix("backbone."): value
        for name, value in source_state.items()
        if name.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"checkpoint contains no backbone weights: {path}")
    model.backbone.load_state_dict(backbone_state, strict=True)
    return {
        "type": "mdlm_pretrained_backbone",
        "path": str(path.resolve()),
        "checkpoint_step": int(payload.get("step", 0)),
        "weights": "ema" if use_ema else "online",
    }


@torch.no_grad()
def _update_ema_state(model, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    parameter_names = {name for name, _ in model.named_parameters()}
    for name, value in model.state_dict().items():
        source = value.detach()
        if name in parameter_names and source.is_floating_point():
            ema_state[name].lerp_(source, 1.0 - decay)
        else:
            ema_state[name].copy_(source)


@contextmanager
def _temporary_model_state(model, state: dict[str, torch.Tensor]):
    original = _clone_model_state(model)
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(original)


@torch.no_grad()
def _search_bd3_schedule(model, corpus, device, batch_size: int, max_batches: int, precision: str):
    candidates = model.schedule_candidates()
    observations: dict[tuple[float, float], list[torch.Tensor]] = {
        candidate: [] for candidate in candidates
    }
    limit = min(max_batches, model.schedule_search_batches)
    for batch in corpus.sequential_batches("validation", batch_size, limit):
        batch = batch.to(device)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        next_cpu_rng = None
        next_cuda_rng = None
        for index, candidate in enumerate(candidates):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            with _autocast(device, precision):
                metrics = model.compute_loss(batch, mask_rate_bounds=candidate)
            observations[candidate].append(metrics["per_block_nelbo"].float().cpu())
            if index == 0:
                next_cpu_rng = torch.get_rng_state()
                if cuda_rng is not None:
                    next_cuda_rng = torch.cuda.get_rng_state(device)
        assert next_cpu_rng is not None
        torch.set_rng_state(next_cpu_rng)
        if next_cuda_rng is not None:
            torch.cuda.set_rng_state(next_cuda_rng, device)

    variances: dict[tuple[float, float], float] = {}
    for candidate, values in observations.items():
        # The official search compares the sum of within-validation-batch
        # variances.  Keeping batches separate avoids adding variance caused by
        # different examples appearing in different batches.
        batch_variances = [
            value.reshape(-1).var(unbiased=value.numel() > 1) for value in values
        ]
        variances[candidate] = float(torch.stack(batch_variances).sum().item())
    best = min(variances, key=lambda candidate: (variances[candidate], candidate))
    model.set_mask_rate_bounds(*best)
    return best, variances[best], len(candidates)


@torch.no_grad()
def validate(model, corpus, device, batch_size: int, max_batches: int, precision: str) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for batch in corpus.sequential_batches("validation", batch_size, max_batches):
        batch = batch.to(device)
        with _autocast(device, precision):
            if model.model_type == "bd3lm":
                metrics = model.compute_loss(
                    batch, mask_rate_bounds=(float(model.min_time), 1.0)
                )
            else:
                metrics = model.compute_loss(batch)
        token_count = float(metrics["token_count"].item())
        for key, value in metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                if key == "token_count":
                    totals[key] = totals.get(key, 0.0) + float(value.item())
                else:
                    totals[key] = totals.get(key, 0.0) + float(value.item()) * token_count
                    weights[key] = weights.get(key, 0.0) + token_count
    if not totals:
        raise RuntimeError("validation split produced no batches")
    results = {
        key: value if key == "token_count" else value / weights[key]
        for key, value in totals.items()
    }
    if model.model_type == "bd3lm" and model.schedule_search:
        best, variance, candidate_count = _search_bd3_schedule(
            model, corpus, device, batch_size, max_batches, precision
        )
        results.update(
            selected_mask_rate_min=best[0],
            selected_mask_rate_max=best[1],
            selected_schedule_nelbo_variance=variance,
            schedule_candidates=float(candidate_count),
        )
    model.train()
    return results


def train(config: dict[str, Any], run_dir: Path, resume: str) -> None:
    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    resume_path = last_path if resume == "auto" else Path(resume) if resume != "none" else None
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(str(tensorboard_dir)) if SummaryWriter is not None else None

    corpus = create_corpus(config)
    (run_dir / "dataset_metadata.json").write_text(json.dumps(corpus.metadata(), indent=2) + "\n")
    model = create_model(config).to(device)
    # Lower is better for both monitored quantities; AR reports exact NLL and the
    # diffusion models report their epsilon-truncated NELBO estimate.
    monitor_key = "nll" if model.model_type == "ar" else "nelbo"
    initialization: dict[str, Any] = {"type": "scratch"}
    pretrained = os.environ.get("BD3_PRETRAIN_CHECKPOINT") or training.get("from_pretrained")
    resume_exists = resume_path is not None and resume_path.exists()
    if training.get("require_pretrained", False) and pretrained is None and not resume_exists:
        raise ValueError(
            "this BD3-LM config requires an MDLM checkpoint: set "
            "training.from_pretrained or BD3_PRETRAIN_CHECKPOINT"
        )
    if pretrained is not None and not resume_exists:
        initialization = _load_pretrained_backbone(model, pretrained, device)
        print(f"Initialized BD3-LM backbone from {pretrained}")
    counts = model.num_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=tuple(training.get("betas", (0.9, 0.95))),
        eps=training.get("epsilon", 1e-8),
        weight_decay=training["weight_decay"],
    )
    scheduler = _scheduler(optimizer, training["warmup_steps"], training["max_steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and training["precision"] == "fp16")
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    step = 0
    processed_tokens = 0
    elapsed_before_resume = 0.0
    best_metric: float | None = None
    resume_payload = None
    if resume_exists:
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        load_model_state(model, resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler.load_state_dict(resume_payload["scaler"])
        step = int(resume_payload["step"])
        processed_tokens = int(resume_payload["processed_tokens"])
        best_metric = resume_payload.get("best_metric")
        elapsed_before_resume = float(resume_payload.get("wall_clock_seconds", 0.0))
        batch_generator.set_state(resume_payload["batch_generator_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        initialization = {"type": "resume", "path": str(resume_path.resolve())}
        print(f"Resumed {resume_path} at optimizer step {step}")

    ema_decay = float(training.get("ema_decay", 0.9999))
    if not 0 <= ema_decay < 1:
        raise ValueError("training.ema_decay must be in [0, 1)")
    ema_state = None
    if ema_decay > 0:
        if resume_payload is not None and resume_payload.get("ema_model") is not None:
            ema_state = {
                name: value.detach().clone()
                for name, value in resume_payload["ema_model"].items()
            }
        else:
            ema_state = _clone_model_state(model)

    metadata = {
        "model_type": model.model_type,
        "parameters": counts,
        "dataset": config["dataset"].get("name", "text8"),
        "tokenizer": corpus.metadata().get("tokenizer", "character"),
        "vocabulary_size": corpus.vocab_size,
        "sequence_length": corpus.sequence_length,
        "seed": seed,
        "optimizer": "AdamW",
        "learning_rate": training["learning_rate"],
        "scheduler": "linear_warmup_cosine_decay",
        "precision": training["precision"],
        "ema_decay": ema_decay,
        "evaluation_weights": "ema" if ema_state is not None else "online",
        "global_batch_size": training["batch_size"] * training["gradient_accumulation_steps"],
        "gpu_type": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _git_commit(),
        "checkpoint_path": str(last_path.resolve()),
        "best_checkpoint_path": str(best_path.resolve()),
        "monitored_metric": monitor_key,
        "tensorboard_logdir": str(tensorboard_dir.resolve()),
        "initialization": initialization,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    started = time.perf_counter()

    def _snapshot() -> dict[str, Any]:
        """Full, resumable checkpoint payload at the current step."""
        return {
            "format_version": 2,
            "model": model.state_dict(),
            "ema_model": ema_state,
            "ema_decay": ema_decay,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "processed_tokens": processed_tokens,
            "wall_clock_seconds": elapsed_before_resume + time.perf_counter() - started,
            "config": config,
            "batch_generator_state": batch_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "parameter_counts": counts,
            "initialization": initialization,
            "monitored_metric": monitor_key,
            "best_metric": best_metric,
        }

    model.train()
    optimizer.zero_grad(set_to_none=True)
    max_steps = int(training["max_steps"])
    accumulation = int(training["gradient_accumulation_steps"])
    while step < max_steps:
        step_metrics: dict[str, float] = {}
        for _ in range(accumulation):
            batch = corpus.random_batch("train", training["batch_size"], batch_generator, device=device)
            with _autocast(device, training["precision"]):
                metrics = model.compute_loss(batch)
                loss = metrics["loss"] / accumulation
            scaler.scale(loss).backward()
            processed_tokens += batch.numel()
            for key, value in metrics.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    step_metrics[key] = step_metrics.get(key, 0.0) + float(value.item()) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        if ema_state is not None:
            _update_ema_state(model, ema_state, ema_decay)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        if step % training.get("log_every", 10) == 0 or step == 1:
            print(json.dumps({"step": step, "processed_tokens": processed_tokens, **step_metrics}))
            if writer is not None:
                for key, value in step_metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
                writer.add_scalar("train/processed_tokens", processed_tokens, step)
        if step % training["validation_every"] == 0 or step == max_steps:
            validation_state = (
                _temporary_model_state(model, ema_state)
                if ema_state is not None
                else nullcontext()
            )
            with validation_state:
                values = validate(
                    model,
                    corpus,
                    device,
                    training["evaluation_batch_size"],
                    training["validation_batches"],
                    training["precision"],
                )
            if model.model_type == "bd3lm" and "selected_mask_rate_min" in values:
                model.set_mask_rate_bounds(
                    values["selected_mask_rate_min"], values["selected_mask_rate_max"]
                )
                if ema_state is not None:
                    ema_state["mask_rate_min"].fill_(values["selected_mask_rate_min"])
                    ema_state["mask_rate_max"].fill_(values["selected_mask_rate_max"])
            with (run_dir / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **values}) + "\n")
            print(json.dumps({"validation_step": step, **values}))
            if writer is not None:
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"val/{key}", float(value), step)
            monitored = values.get(monitor_key)
            if monitored is not None and (best_metric is None or monitored < best_metric):
                best_metric = float(monitored)
                _save_checkpoint(best_path, _snapshot())
                metadata.update(best_metric=best_metric, best_metric_step=step)
                print(json.dumps({"best_checkpoint_step": step, monitor_key: best_metric}))
        if step % training["checkpoint_every"] == 0 or step == max_steps:
            state = _snapshot()
            wall_clock = state["wall_clock_seconds"]
            _save_checkpoint(last_path, state)
            if step == max_steps:
                _save_checkpoint(checkpoint_dir / f"step_{step:08d}.pt", state)
            metadata.update(
                training_steps=step,
                total_processed_tokens=processed_tokens,
                wall_clock_seconds=wall_clock,
            )
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if writer is not None:
        writer.flush()
        writer.close()


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``--set dotted.key=value`` overrides onto a loaded config in place."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"--set target {key!r} is not an existing config key")
        node[parts[-1]] = _coerce(raw)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set training.batch_size=64",
    )
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    train(config, args.run_dir, args.resume)


if __name__ == "__main__":
    main()
