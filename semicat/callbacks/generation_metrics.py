"""Periodic NFE text generation plus MAUVE and external-LM perplexity."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


SCHEMA = "bcfm-textdump/v1"


class GenerationMetricsCallback(Callback):
    """Evaluate a live text CFM at a fixed NFE without perturbing train RNG.

    Sampling is done in-process because the live state may not have a permanent
    checkpoint. External judge models are loaded in a child process, so all of
    their CUDA allocations are released before training resumes.
    """

    def __init__(
        self,
        output_dir: str,
        project_root: str,
        reference_dump: str,
        reference_feature_cache: str,
        resume_checkpoint: str,
        nfe: int = 32,
        n_samples: int = 512,
        sampling_batch_size: int = 64,
        seed: int = 0,
        every_n_train_steps: int = 10_000,
        judge: str = "EleutherAI/gpt-j-6B",
        judge_batch_size: int = 8,
        judge_dtype: str = "float32",
        featurize_model: str = "gpt2-large",
        context_size: int = 256,
        run_on_first_validation: bool = True,
        run_on_train_end: bool = True,
        failure_policy: Literal["raise", "warn"] = "warn",
    ) -> None:
        super().__init__()
        if nfe < 1 or n_samples < 1 or sampling_batch_size < 1:
            raise ValueError("nfe, n_samples, and sampling_batch_size must be positive")
        if every_n_train_steps < 1:
            raise ValueError("every_n_train_steps must be positive")
        if failure_policy not in {"raise", "warn"}:
            raise ValueError("failure_policy must be 'raise' or 'warn'")

        self.output_dir = Path(output_dir).expanduser()
        self.project_root = Path(project_root).expanduser()
        self.reference_dump = Path(reference_dump).expanduser()
        self.reference_feature_cache = Path(reference_feature_cache).expanduser()
        self.resume_checkpoint = Path(resume_checkpoint).expanduser()
        self.nfe = int(nfe)
        self.n_samples = int(n_samples)
        self.sampling_batch_size = int(sampling_batch_size)
        self.seed = int(seed)
        self.every_n_train_steps = int(every_n_train_steps)
        self.judge = judge
        self.judge_batch_size = int(judge_batch_size)
        self.judge_dtype = judge_dtype
        self.featurize_model = featurize_model
        self.context_size = int(context_size)
        self.run_on_first_validation = bool(run_on_first_validation)
        self.run_on_train_end = bool(run_on_train_end)
        self.failure_policy = failure_policy

        self._last_successful_step: int | None = None
        self._last_attempted_step: int | None = None
        self._successful_evals = 0
        self._failed_evals = 0

    @property
    def state_key(self) -> str:
        return f"{type(self).__qualname__}[nfe={self.nfe},seed={self.seed}]"

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if not self.reference_dump.is_file():
            raise FileNotFoundError(f"MAUVE reference dump not found: {self.reference_dump}")
        if not self.resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {self.resume_checkpoint}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_feature_cache.parent.mkdir(parents=True, exist_ok=True)
        protocol = {
            "schema": "semicat-generation-validation/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "output_dir": str(self.output_dir),
            "project_root": str(self.project_root),
            "resume_checkpoint": str(self.resume_checkpoint),
            "resume_checkpoint_sha256_head": self._sha256_head(self.resume_checkpoint),
            "reference_dump": str(self.reference_dump),
            "reference_dump_sha256": self._sha256(self.reference_dump),
            "reference_feature_cache": str(self.reference_feature_cache),
            "nfe": self.nfe,
            "n_samples": self.n_samples,
            "sampling_batch_size": self.sampling_batch_size,
            "seed": self.seed,
            "every_n_train_steps": self.every_n_train_steps,
            "run_on_first_validation": self.run_on_first_validation,
            "run_on_train_end": self.run_on_train_end,
            "discretize": "argmax",
            "judge": self.judge,
            "judge_batch_size": self.judge_batch_size,
            "judge_dtype": self.judge_dtype,
            "featurize_model": self.featurize_model,
            "context_size": self.context_size,
            "device": str(pl_module.device),
            "gpu": torch.cuda.get_device_name(pl_module.device)
            if pl_module.device.type == "cuda"
            else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_commit": self._git_commit(),
        }
        self._write_json(self.output_dir / "protocol.json", protocol)
        rank_zero_info(
            "Generation metrics enabled: "
            f"NFE={self.nfe}, samples={self.n_samples}, seed={self.seed}, "
            f"cadence={self.every_n_train_steps} train steps, "
            f"judge={self.judge}, MAUVE={self.featurize_model}, "
            f"reference={self.reference_dump}, output={self.output_dir}"
        )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (
            trainer.sanity_checking
            or trainer.state.fn != TrainerFn.FITTING
            or not trainer.is_global_zero
        ):
            return
        step = int(trainer.global_step)
        if step == self._last_attempted_step:
            return
        if self._last_successful_step is None:
            if not self.run_on_first_validation:
                self._last_successful_step = step
                return
        elif step - self._last_successful_step < self.every_n_train_steps:
            return
        self._run_or_handle(trainer, pl_module, step, reason="validation")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if (
            self.run_on_train_end
            and step > 0
            and step != self._last_successful_step
            and step != self._last_attempted_step
        ):
            self._run_or_handle(trainer, pl_module, step, reason="train_end")
        rank_zero_info(
            "Generation metrics complete: "
            f"successful={self._successful_evals}, failed={self._failed_evals}, "
            f"last_successful_step={self._last_successful_step}, "
            f"summary={self.output_dir / 'summary.json'}"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_successful_step": self._last_successful_step,
            "last_attempted_step": self._last_attempted_step,
            "successful_evals": self._successful_evals,
            "failed_evals": self._failed_evals,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._last_successful_step = state_dict.get("last_successful_step")
        self._last_attempted_step = state_dict.get("last_attempted_step")
        self._successful_evals = int(state_dict.get("successful_evals", 0))
        self._failed_evals = int(state_dict.get("failed_evals", 0))

    def _run_or_handle(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step: int,
        reason: str,
    ) -> None:
        self._last_attempted_step = step
        try:
            metrics = self._evaluate(trainer, pl_module, step, reason)
        except Exception as error:
            self._failed_evals += 1
            failure = {
                "status": "failed",
                "global_step": step,
                "epoch": int(trainer.current_epoch),
                "reason": reason,
                "error_type": type(error).__name__,
                "error": str(error),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            step_dir = self.output_dir / f"step={step:08d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(step_dir / "error.json", failure)
            self._append_summary(failure)
            if self.failure_policy == "raise":
                raise
            rank_zero_warn(
                f"Generation metrics failed at step={step}; training will continue: {error!r}"
            )
            return

        self._last_successful_step = step
        self._successful_evals += 1
        self._append_summary(metrics)
        logger_metrics = {
            f"val/mauve_nfe{self.nfe}": metrics["mauve"],
            f"val/gen_ppl_nfe{self.nfe}": metrics["gen_ppl"],
            f"val/frontier_integral_nfe{self.nfe}": metrics["frontier_integral"],
            f"val/sampling_seconds_nfe{self.nfe}": metrics["sampling_seconds"],
            f"val/scoring_seconds_nfe{self.nfe}": metrics["scoring_seconds"],
        }
        for logger in trainer.loggers:
            logger.log_metrics(logger_metrics, step=step)
        rank_zero_info(
            "Generation metric result: "
            f"step={step}, NFE={self.nfe}, MAUVE={metrics['mauve']:.8f}, "
            f"genPPL={metrics['gen_ppl']:.6f}, "
            f"sampling_seconds={metrics['sampling_seconds']:.1f}, "
            f"scoring_seconds={metrics['scoring_seconds']:.1f}, "
            f"artifacts={metrics['step_dir']}"
        )

    def _evaluate(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step: int,
        reason: str,
    ) -> dict[str, Any]:
        step_dir = self.output_dir / f"step={step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        dump_path = step_dir / f"samples_nfe{self.nfe}.json"
        token_path = dump_path.with_suffix(".tokens.npz")
        score_path = step_dir / "scores.json"

        if dump_path.is_file() and token_path.is_file():
            rank_zero_info(f"Generation cache hit: reusing {dump_path} and {token_path}")
            payload = json.loads(dump_path.read_text())
            sampling_seconds = float(payload["points"][0]["sampling_seconds"])
        else:
            ids, sampling_seconds = self._generate(pl_module, step)
            rank_zero_info(
                f"Pipeline stage=detokenize step={step} samples={ids.shape[0]}"
            )
            texts = trainer.datamodule.tensor_to_strings(ids)
            point_id = f"argmax_nfe{self.nfe}"
            np.savez_compressed(token_path, **{point_id: ids.numpy().astype(np.int32)})
            payload = {
                "schema": SCHEMA,
                "label": f"train_step{step}_full_cfm_nfe{self.nfe}",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": self._git_commit(),
                "model": {
                    "kind": "live_training_state",
                    "checkpoint": str(self.resume_checkpoint),
                    "checkpoint_step": step,
                    "dataset": "tinystories",
                    "tokenizer": "gpt2",
                    "arch": self._architecture(pl_module),
                },
                "sampler": {
                    "kind": "full_cfm",
                    "block_size": None,
                    "length": int(pl_module.in_shape[0]),
                    "seed": self.seed,
                },
                "points": [
                    {
                        "id": point_id,
                        "nfe": self.nfe,
                        "discretize": "argmax",
                        "forwards_per_sequence": self.nfe,
                        "n_samples": int(ids.shape[0]),
                        "sampling_seconds": round(sampling_seconds, 3),
                        "texts": texts,
                    }
                ],
            }
            self._write_json(dump_path, payload)
            (step_dir / "samples_preview.txt").write_text("\n\n".join(texts[:32]) + "\n")
            rank_zero_info(
                f"Pipeline stage=samples_written step={step} dump={dump_path} "
                f"tokens={token_path} preview={step_dir / 'samples_preview.txt'}"
            )

        if score_path.is_file():
            rank_zero_info(f"Metric cache hit: reusing {score_path}")
        else:
            command = [
                sys.executable,
                "-m",
                "eval.score_texts",
                "--dumps",
                str(dump_path),
                "--reference",
                str(self.reference_dump),
                "--metrics",
                "mauve",
                "gen_ppl",
                "--judge",
                self.judge,
                "--judge-batch-size",
                str(self.judge_batch_size),
                "--judge-dtype",
                self.judge_dtype,
                "--featurize-model",
                self.featurize_model,
                "--mauve-reference-cache",
                str(self.reference_feature_cache),
                "--context-size",
                str(self.context_size),
                "--device",
                str(pl_module.device),
                "--device-id",
                str(pl_module.device.index or 0),
                "--out",
                str(score_path),
            ]
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            rank_zero_info(
                f"Pipeline stage=external_scoring step={step} command={' '.join(command)}"
            )
            subprocess.run(command, cwd=self.project_root, env=env, check=True)

        scores = json.loads(score_path.read_text())
        rows = scores.get("rows", [])
        if len(rows) != 1:
            raise ValueError(f"expected exactly one score row in {score_path}, got {len(rows)}")
        row = rows[0]
        for key in ("mauve", "frontier_integral", "gen_ppl", "scoring_seconds"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math_is_finite(value):
                raise ValueError(f"invalid {key}={value!r} in {score_path}")
        result = {
            "status": "complete",
            "global_step": step,
            "epoch": int(trainer.current_epoch),
            "reason": reason,
            "nfe": self.nfe,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "mauve": float(row["mauve"]),
            "frontier_integral": float(row["frontier_integral"]),
            "gen_ppl": float(row["gen_ppl"]),
            "sampling_seconds": sampling_seconds,
            "scoring_seconds": float(row["scoring_seconds"]),
            "dump": str(dump_path),
            "tokens": str(token_path),
            "scores": str(score_path),
            "step_dir": str(step_dir),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write_json(step_dir / "metrics.json", result)
        return result

    @torch.inference_mode()
    def _generate(
        self, pl_module: LightningModule, step: int
    ) -> tuple[torch.Tensor, float]:
        device = pl_module.device
        was_training = pl_module.training
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            torch.cuda.empty_cache()

        outputs = []
        done = 0
        started = time.perf_counter()
        try:
            pl_module.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(self.seed)
                batches = (self.n_samples + self.sampling_batch_size - 1) // self.sampling_batch_size
                for batch_index in range(batches):
                    size = min(self.sampling_batch_size, self.n_samples - done)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        endpoint = pl_module.sample_flow_map_batch(
                            batch_size=size,
                            sampling_steps=self.nfe,
                        )
                    outputs.append(endpoint.argmax(dim=-1).cpu())
                    done += size
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    rank_zero_info(
                        "Pipeline stage=generation "
                        f"step={step} NFE={self.nfe} batch={batch_index + 1}/{batches} "
                        f"samples={done}/{self.n_samples} elapsed={elapsed:.1f}s "
                        f"samples_per_s={done / max(elapsed, 1e-9):.2f}"
                    )
                    del endpoint
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if was_training:
                pl_module.train()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(outputs, dim=0)[: self.n_samples], time.perf_counter() - started

    def _append_summary(self, row: dict[str, Any]) -> None:
        path = self.output_dir / "summary.json"
        if path.is_file():
            payload = json.loads(path.read_text())
        else:
            payload = {"schema": "semicat-generation-validation/v1", "rows": []}
        payload["rows"].append(row)
        self._write_json(path, payload)

    @staticmethod
    def _architecture(pl_module: LightningModule) -> dict[str, Any]:
        net = pl_module.net
        return {
            "vocab_size": int(net.vocab_size),
            "hidden_size": int(net.output_layer.linear.in_features),
            "length": int(pl_module.in_shape[0]),
            "n_heads": int(net.blocks[0].n_heads),
            "n_blocks": len(net.blocks),
            "cond_dim": int(net.s_map.mlp[0].out_features),
            "embed_type": "rms" if hasattr(net.vocab_embed, "final_norm") else "naive",
            "jvp_attention_backend": getattr(net, "jvp_attention_backend", None),
        }

    def _git_commit(self) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.project_root,
                text=True,
            ).strip()
        except Exception:
            return None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_head(path: Path, limit: int = 64 << 20) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(limit))
        return digest.hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def math_is_finite(value: int | float) -> bool:
    return bool(np.isfinite(float(value)))
