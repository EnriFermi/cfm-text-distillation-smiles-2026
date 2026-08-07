#!/usr/bin/env python3
"""Mirror a live Lightning CSV log into one persistent Comet experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import yaml
from comet_ml import ExistingExperiment, Experiment
from dotenv import load_dotenv


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        result[prefix] = json.dumps(value)
    elif value is None or isinstance(value, (str, int, float, bool)):
        result[prefix] = value
    else:
        result[prefix] = str(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _sync_csv(
    experiment,
    csv_path: Path,
    minimum_step: int,
    last_steps: dict[str, int],
) -> int:
    if not csv_path.is_file():
        return 0
    logged = 0
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw_step = row.get("step")
        if not raw_step:
            continue
        try:
            step = int(float(raw_step))
        except ValueError:
            continue
        # Lightning's per-validation-batch rows use local steps 0..49. The
        # epoch aggregate and all train rows carry the true global step.
        if step < minimum_step:
            continue
        metrics: dict[str, float] = {}
        for name, raw_value in row.items():
            if name in {"step", "epoch"} or not raw_value:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if not math.isfinite(value) or step <= int(last_steps.get(name, -1)):
                continue
            metrics[name] = value
        if not metrics:
            continue
        raw_epoch = row.get("epoch")
        epoch = int(float(raw_epoch)) if raw_epoch else None
        experiment.log_metrics(metrics, step=step, epoch=epoch)
        for name in metrics:
            last_steps[name] = step
        logged += len(metrics)
    return logged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--minimum-step", type=int, default=75001)
    parser.add_argument("--source-checkpoint-step", type=int, default=75001)
    parser.add_argument("--target-step", type=int, default=200000)
    parser.add_argument("--project", default="bcfm")
    parser.add_argument("--name", required=True)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    workspace = os.environ.get("COMET_WORKSPACE")
    api_key = os.environ.get("COMET_API_KEY")
    if not workspace or not api_key:
        raise RuntimeError("COMET_API_KEY and COMET_WORKSPACE must be set")

    state_path = run_dir / "comet_mirror.json"
    state = _read_json(state_path)
    common = dict(
        api_key=api_key,
        project_name=args.project,
        workspace=workspace,
        log_code=False,
        log_graph=False,
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_output_logging=None,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        auto_log_co2=False,
    )
    if state.get("experiment_key"):
        experiment = ExistingExperiment(
            previous_experiment=state["experiment_key"], **common
        )
    else:
        experiment = Experiment(**common)
        experiment.set_name(args.name)
        tags = args.tags or [
            "M1",
            "cfm",
            "tinystories",
            f"resume-{args.source_checkpoint_step}",
            f"target-{args.target_step}",
            "nfe32",
        ]
        experiment.add_tags(tags)
        config_path = run_dir / ".hydra" / "config.yaml"
        if config_path.is_file():
            experiment.log_parameters(_flatten(yaml.safe_load(config_path.read_text())))
            experiment.log_asset(str(config_path), file_name="resolved_config.yaml")
        for asset in (
            run_dir / "launch_manifest.txt",
            run_dir / "generation_metrics" / "protocol.json",
        ):
            if asset.is_file():
                experiment.log_asset(str(asset))
        experiment.log_others(
            {
                "source_checkpoint_global_step": args.source_checkpoint_step,
                "target_global_step": args.target_step,
                "logger_source": "live Lightning CSV mirror",
                "run_dir": str(run_dir),
            }
        )
        state = {
            "experiment_key": experiment.get_key(),
            "url": experiment.url,
            "last_steps": {},
        }
        _write_json(state_path, state)

    print(f"COMET_URL={experiment.url}", flush=True)
    csv_path = run_dir / "csv" / "metrics.csv"
    try:
        while True:
            logged = _sync_csv(
                experiment,
                csv_path,
                args.minimum_step,
                state.setdefault("last_steps", {}),
            )
            state["last_sync_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            state["training_pid"] = args.training_pid
            state["training_alive"] = _process_alive(args.training_pid)
            _write_json(state_path, state)
            print(
                f"sync logged_metrics={logged} training_alive={state['training_alive']}",
                flush=True,
            )
            if not state["training_alive"]:
                break
            time.sleep(args.poll_seconds)
    finally:
        experiment.end()


if __name__ == "__main__":
    main()
