#!/usr/bin/env python3
"""Export and analyze train/validation loss dynamics for the active BCFM run."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import API
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
EXPERIMENT_KEY = "cef17582c0174d4a99bdb558a304f890"
RUN_DIR = ROOT / "logs/train/2026-07-18_10-46-57_12345_local"
METRICS = [
    "val/loss_epoch",
    "val/loss_step",
    "val/vf_loss",
    "val/sd_loss",
    "train/loss_epoch",
    "train/loss_step",
    "train/vf_loss",
    "train/sd_loss",
]


def numeric(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        item = dict(row)
        item["metricValue"] = float(item["metricValue"])
        item["timestamp"] = int(item["timestamp"])
        item["step"] = int(item["step"])
        item["epoch"] = int(item["epoch"])
        result.append(item)
    return result


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "sem_naive": float(x.std(ddof=1) / math.sqrt(len(x))),
        "min": float(x.min()),
        "max": float(x.max()),
        "n_batches": int(len(x)),
    }


cfg = OmegaConf.load(ROOT / "configs/logger/comet.yaml")
experiment = API(api_key=cfg.comet.api_key).get_experiment_by_key(EXPERIMENT_KEY)
raw = {name: numeric(experiment.get_metrics(name)) for name in METRICS}
(OUTPUT / "raw_metrics.json").write_text(json.dumps(raw, indent=2))

val_by_metric_epoch: dict[str, dict[int, list[dict]]] = {}
for metric in ("val/loss_step", "val/vf_loss", "val/sd_loss"):
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in raw[metric]:
        grouped[row["epoch"]].append(row)
    val_by_metric_epoch[metric] = grouped

train_epoch_rows = raw["train/loss_epoch"]
train_vf_rows = raw["train/vf_loss"]
train_sd_rows = raw["train/sd_loss"]
events: list[dict] = []
for epoch_loss in raw["val/loss_epoch"]:
    epoch = epoch_loss["epoch"]
    step = epoch_loss["step"] + 1
    vf = stats(
        [row["metricValue"] for row in val_by_metric_epoch["val/vf_loss"][epoch]]
    )
    sd = stats(
        [row["metricValue"] for row in val_by_metric_epoch["val/sd_loss"][epoch]]
    )
    total = stats(
        [row["metricValue"] for row in val_by_metric_epoch["val/loss_step"][epoch]]
    )
    recent_train_vf = [
        row["metricValue"] for row in train_vf_rows if step - 1000 <= row["step"] < step
    ]
    recent_train_sd = [
        row["metricValue"] for row in train_sd_rows if step - 1000 <= row["step"] < step
    ]
    preceding_train_epoch = max(
        (row for row in train_epoch_rows if row["step"] < step),
        key=lambda row: row["step"],
    )
    events.append(
        {
            "global_step": step,
            "epoch": epoch,
            "val_loss_epoch": epoch_loss["metricValue"],
            "val_total_batch_mean": total["mean"],
            "val_total_batch_std": total["std"],
            "val_total_naive_sem": total["sem_naive"],
            "val_vf_mean": vf["mean"],
            "val_vf_std": vf["std"],
            "val_vf_naive_sem": vf["sem_naive"],
            "val_sd_mean": sd["mean"],
            "val_sd_std": sd["std"],
            "val_sd_naive_sem": sd["sem_naive"],
            "val_batches": total["n_batches"],
            "preceding_train_epoch_step": preceding_train_epoch["step"],
            "preceding_train_loss_epoch": preceding_train_epoch["metricValue"],
            "recent_train_vf_mean_1k": float(np.mean(recent_train_vf)),
            "recent_train_sd_mean_1k": float(np.mean(recent_train_sd)),
        }
    )

with (OUTPUT / "val_events.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(events[0]))
    writer.writeheader()
    writer.writerows(events)
(OUTPUT / "val_events.json").write_text(json.dumps(events, indent=2))

resolved = OmegaConf.load(RUN_DIR / ".hydra/config.yaml")
k = int(resolved.data.k)
split = [int(x) for x in resolved.data.train_val_test_split]
train_tokens = (ROOT / "data/text8/train.bin").stat().st_size // 2
holdout_tokens = (ROOT / "data/text8/val.bin").stat().st_size // 2
geometry = {
    "k": k,
    "configured_train_val_test_split": split,
    "train_bin_tokens": train_tokens,
    "holdout_bin_tokens": holdout_tokens,
    "actual_train_raw_tokens_used": split[0],
    "actual_val_raw_tokens_used": split[1],
    "actual_test_raw_tokens_used": split[2],
    "actual_train_windows_stride1": split[0] - k + 1,
    "actual_val_windows_stride1": split[1] - k + 1,
    "actual_test_windows_stride1": split[2] - k + 1,
    "train_corpus_coverage_fraction": split[0] / train_tokens,
    "holdout_corpus_coverage_fraction": (split[1] + split[2]) / holdout_tokens,
    "train_window_adjacent_overlap_tokens": k - 1,
    "validation_shuffle": False,
    "validation_batches": math.ceil((split[1] - k + 1) / resolved.data.batch_size),
    "batch_size": int(resolved.data.batch_size),
    "sd_prop": float(resolved.model.sd_prop),
    "block_size": int(resolved.model.block_size),
    "sd_type": str(resolved.model.sd_type),
    "learning_rate": float(resolved.model.optimizer.lr),
    "scheduler": resolved.model.scheduler,
}
(OUTPUT / "data_geometry.json").write_text(json.dumps(geometry, indent=2))

first, last = events[0], events[-1]
delta_total = last["val_loss_epoch"] - first["val_loss_epoch"]
delta_vf = last["val_vf_mean"] - first["val_vf_mean"]
delta_sd = last["val_sd_mean"] - first["val_sd_mean"]
evidence = {
    "experiment_key": EXPERIMENT_KEY,
    "experiment_name": experiment.get_name(),
    "run_dir": str(RUN_DIR),
    "uniform_27way_cross_entropy": math.log(27),
    "first_validation": first,
    "last_validation": last,
    "delta_val_total": delta_total,
    "delta_val_vf": delta_vf,
    "delta_val_sd": delta_sd,
    "fraction_of_total_growth_from_vf": delta_vf / delta_total,
    "first_to_last_change_in_naive_sem_units": delta_total
    / math.sqrt(
        first["val_total_naive_sem"] ** 2 + last["val_total_naive_sem"] ** 2
    ),
    "note_on_sem": (
        "Naive batch SEM understates uncertainty because stride-1 Text8 windows "
        "are correlated; it is reported only as a scale check."
    ),
}
(OUTPUT / "mechanism_evidence.json").write_text(json.dumps(evidence, indent=2))

train_steps = [row["step"] + 1 for row in train_epoch_rows]
train_loss = [row["metricValue"] for row in train_epoch_rows]
val_steps = [row["global_step"] for row in events]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].plot(train_steps, train_loss, marker="o", markersize=3, color="#0072B2")
axes[0].set_title("Training objective")
axes[0].set_xlabel("Optimizer step")
axes[0].set_ylabel("train/loss_epoch")

axes[1].plot(
    val_steps,
    [row["val_loss_epoch"] for row in events],
    marker="o",
    label="total",
    color="#D55E00",
)
axes[1].plot(
    val_steps,
    [row["val_vf_mean"] for row in events],
    marker="o",
    label="VF cross-entropy",
    color="#CC79A7",
)
axes[1].axhline(
    math.log(27), linestyle="--", color="#555555", label="uniform CE = ln(27)"
)
axes[1].set_title("Validation growth is VF-dominated")
axes[1].set_xlabel("Optimizer step")
axes[1].set_ylabel("Validation loss")
axes[1].legend(fontsize=8)

axes[2].plot(
    val_steps,
    [row["val_sd_mean"] for row in events],
    marker="o",
    label="validation SD",
    color="#009E73",
)
axes[2].plot(
    val_steps,
    [row["recent_train_sd_mean_1k"] for row in events],
    marker="o",
    label="training SD (recent 1k)",
    color="#56B4E9",
)
axes[2].set_title("Lagrangian SD component")
axes[2].set_xlabel("Optimizer step")
axes[2].set_ylabel("SD loss")
axes[2].legend(fontsize=8)

for axis in axes:
    axis.grid(alpha=0.25)
fig.suptitle(
    "Blockwise CFM run: train improves while held-out VF cross-entropy degrades",
    fontsize=14,
)
fig.tight_layout()
fig.savefig(OUTPUT / "loss_dynamics.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(json.dumps(evidence, indent=2))
print(f"Wrote {OUTPUT / 'val_events.csv'}")
print(f"Wrote {OUTPUT / 'loss_dynamics.png'}")
