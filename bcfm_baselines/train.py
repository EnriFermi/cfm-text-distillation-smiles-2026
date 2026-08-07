"""Non-interactive Text8 trainer designed for Slurm execution and resumption."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from collections import deque
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
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
from bcfm_baselines.generative_eval import (
    GenerativeEvalConfig,
    evaluate_generation,
    require_metric_dependencies,
    resolve_config as resolve_generative_config,
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_fingerprint() -> str:
    """Hash runnable source/config files even when this folder is not a git repo."""
    root = Path(__file__).resolve().parents[1]
    files = [*root.joinpath("bcfm_baselines").rglob("*.py")]
    files.extend(root.joinpath("configs").rglob("*.yaml"))
    files.extend(root / name for name in ("pyproject.toml", "requirements-a100.txt"))
    digest = hashlib.sha256()
    for path in sorted((path for path in files if path.exists()), key=lambda item: str(item)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "numpy", "transformers", "mauve-text", "scikit-learn", "faiss-cpu"):
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


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


def _scheduler(
    optimizer,
    warmup_steps: int,
    max_steps: int,
    decay: str = "constant",
    warmup_start_factor: float = 1e-3,
):
    """Linear warmup, then either a flat LR or cosine decay.

    ``constant`` is the default because it is what the CFM reference does: its
    ``LinearLR(start_factor=1e-3, total_iters=warmup)`` reaches factor 1.0 and
    stays there for the rest of training. A cosine tail would hand the baselines
    a different effective learning-rate budget than the method they are compared
    against.
    """
    if decay not in ("constant", "cosine"):
        raise ValueError(f"training.lr_decay must be constant or cosine, got {decay!r}")

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return warmup_start_factor + (1.0 - warmup_start_factor) * step / warmup_steps
        if decay == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _clone_model_state(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _nested_value(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _pretrained_config_mismatches(
    source: dict[str, Any], expected: dict[str, Any] | None
) -> list[str]:
    """Describe semantic MDLM/BD3 incompatibilities not visible from tensor shapes."""
    if expected is None:
        return []
    fields = (
        "dataset.name",
        "dataset.tokenizer",
        "dataset.vocabulary_size",
        "dataset.sequence_length",
        "model.data_vocab_size",
        "model.mask_token_id",
        "model.ignore_bos",
        "model.time_conditioning",
    )
    fields += tuple(
        f"model.backbone.{name}" for name in expected.get("model", {}).get("backbone", {})
    )
    mismatches = []
    for field in fields:
        source_value = _nested_value(source, field)
        expected_value = _nested_value(expected, field)
        if source_value != expected_value:
            mismatches.append(f"{field}: checkpoint={source_value!r}, expected={expected_value!r}")
    return mismatches


def _load_pretrained_backbone(
    model,
    checkpoint: str | Path,
    device: torch.device,
    *,
    require_ema: bool = False,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if require_ema and not use_ema:
        raise ValueError(
            "paper-style BD3-LM initialization requires an MDLM EMA checkpoint, "
            f"but {path} contains online weights only; retrain MDLM with "
            "training.ema_decay=0.9999"
        )
    source_state = payload["ema_model"] if use_ema else payload["model"]
    backbone_state = {
        name.removeprefix("backbone."): value
        for name, value in source_state.items()
        if name.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"checkpoint contains no backbone weights: {path}")

    target_state = model.backbone.state_dict()
    mismatches = _pretrained_config_mismatches(payload.get("config", {}), expected_config)
    missing = sorted(set(target_state) - set(backbone_state))
    unexpected = sorted(set(backbone_state) - set(target_state))
    shape_mismatches = [
        f"{name}: checkpoint={tuple(backbone_state[name].shape)}, "
        f"expected={tuple(target_state[name].shape)}"
        for name in sorted(set(target_state) & set(backbone_state))
        if backbone_state[name].shape != target_state[name].shape
    ]
    if missing:
        mismatches.append(f"missing backbone tensors: {missing[:8]}")
    if unexpected:
        mismatches.append(f"unexpected backbone tensors: {unexpected[:8]}")
    mismatches.extend(shape_mismatches[:8])
    if mismatches:
        details = "\n  - ".join(mismatches[:16])
        extra = len(mismatches) - min(len(mismatches), 16)
        suffix = f"\n  - ... and {extra} more mismatch(es)" if extra else ""
        raise ValueError(
            "pretrained MDLM checkpoint is incompatible with the configured BD3-LM "
            f"model: {path}\n  - {details}{suffix}\n"
            "Use an MDLM checkpoint trained with the same dataset/tokenizer/backbone config."
        )
    model.backbone.load_state_dict(backbone_state, strict=True)
    configured_max_steps = payload.get("config", {}).get("training", {}).get("max_steps")
    return {
        "type": "mdlm_pretrained_backbone",
        "path": str(path.resolve()),
        "checkpoint_step": int(payload.get("step", 0)),
        "checkpoint_max_steps": (
            int(configured_max_steps) if configured_max_steps is not None else None
        ),
        "source_dataset": payload.get("config", {}).get("dataset", {}).get("name"),
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
    """Swap model/EMA values in place without allocating another whole model.

    During training both the online model and EMA already reside on the GPU.
    Cloning the complete online state before validation temporarily added a
    third model-sized allocation. Pairwise swaps need only one tensor-sized
    scratch buffer at a time and leave ``state`` holding the online values while
    the model is evaluated with EMA values.
    """

    @torch.no_grad()
    def swap() -> None:
        current = model.state_dict()
        if set(current) != set(state):
            missing = sorted(set(current) - set(state))
            unexpected = sorted(set(state) - set(current))
            raise ValueError(
                "temporary model state is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        incompatible = [
            name
            for name, value in current.items()
            if value.shape != state[name].shape
            or value.dtype != state[name].dtype
            or value.device != state[name].device
        ]
        if incompatible:
            raise ValueError(
                f"temporary model state has incompatible tensors: {incompatible}"
            )
        for name, value in current.items():
            temporary = value.detach().clone()
            value.copy_(state[name])
            state[name].copy_(temporary)

    swap()
    try:
        yield
    finally:
        swap()


def _validation_batches(corpus, batch_size: int, max_batches: int, stride: int | None):
    """Use an explicit Text8 stride without imposing it on corpus test doubles."""
    if stride is None:
        return corpus.sequential_batches("validation", batch_size, max_batches)
    return corpus.sequential_batches(
        "validation", batch_size, max_batches, stride=stride
    )


@torch.no_grad()
def _search_bd3_schedule(
    model,
    corpus,
    device,
    batch_size: int,
    max_batches: int,
    precision: str,
    validation_stride: int | None = None,
):
    candidates = model.schedule_candidates()
    observations: dict[tuple[float, float], list[torch.Tensor]] = {
        candidate: [] for candidate in candidates
    }
    limit = min(max_batches, model.schedule_search_batches)
    for batch in _validation_batches(corpus, batch_size, limit, validation_stride):
        batch = batch.to(device)
        for candidate in candidates:
            # The official search evaluates each candidate with an independent
            # diffusion-time/corruption draw. Reusing one RNG state for every
            # interval silently changes the published selection procedure.
            with _autocast(device, precision):
                metrics = model.compute_loss(batch, mask_rate_bounds=candidate)
            observations[candidate].append(metrics["per_block_nelbo"].float().cpu())

    variance_sums: dict[tuple[float, float], float] = {}
    variances: dict[tuple[float, float], float] = {}
    for candidate, values in observations.items():
        # The official search compares the sum of within-validation-batch
        # variances.  Keeping batches separate avoids adding variance caused by
        # different examples appearing in different batches.
        batch_variances = [
            value.reshape(-1).var(unbiased=value.numel() > 1) for value in values
        ]
        stacked = torch.stack(batch_variances)
        variance_sums[candidate] = float(stacked.sum().item())
        # Official TensorBoard logging divides the selection sum by the number
        # of batches. Expose that interpretable mean for paper-style curves.
        variances[candidate] = float(stacked.mean().item())
    best = min(
        variance_sums,
        key=lambda candidate: (variance_sums[candidate], candidate),
    )
    model.set_mask_rate_bounds(*best)
    return best, variances[best], variance_sums[best], variances


@torch.no_grad()
def validate(
    model,
    corpus,
    device,
    batch_size: int,
    max_batches: int,
    precision: str,
    sampling: dict[str, Any] | None = None,
    generative: GenerativeEvalConfig | None = None,
    search_bd3_schedule: bool = True,
    validation_stride: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for batch in _validation_batches(corpus, batch_size, max_batches, validation_stride):
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
    if model.model_type == "bd3lm" and model.schedule_search and search_bd3_schedule:
        best, variance, variance_sum, candidate_variances = _search_bd3_schedule(
            model,
            corpus,
            device,
            batch_size,
            max_batches,
            precision,
            validation_stride,
        )
        results.update(
            selected_mask_rate_min=best[0],
            selected_mask_rate_max=best[1],
            selected_schedule_nelbo_variance=variance,
            selected_schedule_nelbo_variance_sum=variance_sum,
            schedule_candidates=float(len(candidate_variances)),
        )
        for (lower, upper), candidate_variance in candidate_variances.items():
            results[
                f"schedule_variance/{lower:.3f}_{upper:.3f}"
            ] = candidate_variance
    if generative is not None and generative.enabled and sampling is not None:
        # Sampling runs outside autocast: the judge models are loaded in their own
        # precision and the generators already fix their own dtypes.
        results.update(
            evaluate_generation(model, corpus, sampling, generative, device)
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
    # Both match the CFM reference (semicat/models/semicat.py, semicat/train.py):
    # TF32 matmuls and autotuned cudnn kernels, so the two sides run the same
    # numerics rather than differing by a silent precision default.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
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
    generative = resolve_generative_config(config)
    require_metric_dependencies(generative)
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
        initialization = _load_pretrained_backbone(
            model,
            pretrained,
            device,
            require_ema=bool(training.get("require_pretrained_ema", False)),
            expected_config=config,
        )
        print(f"Initialized BD3-LM backbone from {pretrained}")
    counts = model.num_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=tuple(training.get("betas", (0.9, 0.999))),
        eps=training.get("epsilon", 1e-8),
        weight_decay=training["weight_decay"],
    )
    scheduler = _scheduler(
        optimizer,
        training["warmup_steps"],
        training["max_steps"],
        decay=training.get("lr_decay", "constant"),
        warmup_start_factor=training.get("warmup_start_factor", 1e-3),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and training["precision"] == "fp16")
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    step = 0
    processed_tokens = 0
    elapsed_before_resume = 0.0
    best_metric: float | None = None
    resume_payload = None
    if resume_exists:
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if training.get("require_pretrained_ema", False) and resume_payload.get(
            "ema_model"
        ) is None:
            raise ValueError(
                "paper-style BD3-LM resume requires ema_model, but "
                f"{resume_path} is a legacy/non-EMA checkpoint; resume a run "
                "created with training.ema_decay=0.9999"
            )
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
        "betas": list(training.get("betas", (0.9, 0.999))),
        "scheduler": f"linear_warmup_{training.get('lr_decay', 'constant')}",
        "precision": training["precision"],
        "ema_decay": ema_decay,
        "evaluation_weights": "ema" if ema_state is not None else "online",
        "global_batch_size": training["batch_size"] * training["gradient_accumulation_steps"],
        "gpu_type": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_fingerprint(),
        "python_version": sys.version,
        "dependency_versions": _dependency_versions(),
        "checkpoint_path": str(last_path.resolve()),
        "best_checkpoint_path": str(best_path.resolve()),
        "monitored_metric": monitor_key,
        "tensorboard_logdir": str(tensorboard_dir.resolve()),
        "training_curve_path": str((run_dir / "training.jsonl").resolve()),
        "validation_curve_path": str((run_dir / "validation.jsonl").resolve()),
        "initialization": initialization,
        "generative_evaluation": {
            "enabled": generative.enabled,
            "sample_count": generative.sample_count,
            "gen_ppl_model": generative.ppl_model,
            "mauve_model": generative.mauve_model,
            "num_steps": generative.num_steps,
            "steps_per_block": generative.steps_per_block,
            "first_hitting": generative.first_hitting,
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    started = time.perf_counter()
    recent_nelbos: deque[float] = deque(maxlen=100)

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
        if "nelbo" in step_metrics:
            recent_nelbos.append(step_metrics["nelbo"])
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), training["gradient_clip_norm"]
        )
        scaler.step(optimizer)
        scaler.update()
        if ema_state is not None:
            _update_ema_state(model, ema_state, ema_decay)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        if step % training.get("log_every", 10) == 0 or step == 1:
            wall_clock = elapsed_before_resume + time.perf_counter() - started
            training_record = {
                "step": step,
                "processed_tokens": processed_tokens,
                "wall_clock_seconds": wall_clock,
                "tokens_per_second": processed_tokens / max(wall_clock, 1e-9),
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm),
                **step_metrics,
            }
            if model.model_type == "bd3lm":
                training_record.update(
                    mask_rate_min=float(model.mask_rate_min.item()),
                    mask_rate_max=float(model.mask_rate_max.item()),
                )
            if "nelbo" in step_metrics:
                training_record["epsilon_truncated_nelbo_perplexity_estimate"] = math.exp(
                    min(step_metrics["nelbo"], 80.0)
                )
                if len(recent_nelbos) > 1:
                    training_record["nelbo_variance_100"] = float(
                        np.var(tuple(recent_nelbos), ddof=1)
                    )
            with (run_dir / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(training_record) + "\n")
            print(json.dumps(training_record))
            if writer is not None:
                for key, value in step_metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
                for key, value in training_record.items():
                    if key not in step_metrics and isinstance(value, (int, float)):
                        writer.add_scalar(f"train/{key}", value, step)
        if step % training["validation_every"] == 0 or step == max_steps:
            # Persist the just-completed optimizer state before slow external
            # Gen-PPL/MAUVE judges run, so a judge/OOM failure cannot discard an
            # entire validation interval of training.
            _save_checkpoint(last_path, _snapshot())
            schedule_search_start = int(training.get("schedule_search_start_step", 5000))
            if schedule_search_start < 0:
                raise ValueError("training.schedule_search_start_step must be non-negative")
            # The paper starts adaptive clipping after 5K gradient updates and
            # then repeats the grid search at every ordinary validation.
            search_bd3_schedule = step >= schedule_search_start
            if device.type == "cuda":
                # Training activations are dead here, but the caching allocator
                # may still retain them. Release that cache before the very
                # different GPT-2-vocabulary sampling/judge allocation pattern.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
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
                    sampling=config.get("sampling"),
                    generative=generative,
                    search_bd3_schedule=search_bd3_schedule,
                    validation_stride=training.get("validation_stride"),
                )
            if device.type == "cuda":
                values["peak_gpu_memory_gb"] = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                )
                torch.cuda.empty_cache()
            if "nelbo" in values:
                values["epsilon_truncated_nelbo_perplexity_estimate"] = math.exp(
                    min(values["nelbo"], 80.0)
                )
            elif "nll" in values:
                values["perplexity"] = math.exp(min(values["nll"], 80.0))
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
            sample_texts = values.get("sample_texts")
            if sample_texts:
                samples_dir = run_dir / "validation_samples"
                samples_dir.mkdir(exist_ok=True)
                (samples_dir / f"step_{step:08d}.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "sampler": values.get("generation_sampler"),
                            "num_function_evaluations": values.get(
                                "generation_num_function_evaluations"
                            ),
                            "texts": sample_texts,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            if writer is not None:
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"val/{key}", float(value), step)
                if sample_texts:
                    writer.add_text(
                        "val/sample_texts",
                        "\n\n---\n\n".join(sample_texts),
                        step,
                    )
                writer.flush()
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
