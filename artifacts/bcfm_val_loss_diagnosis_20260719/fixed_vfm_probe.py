#!/usr/bin/env python3
"""Fixed-latent VFM probe across BCFM checkpoints.

The same decorrelated train/validation windows, Gaussian prior, and per-block
times are reused for every checkpoint. This separates weight drift from the
fresh validation RNG used by Lightning.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTPUT = Path(__file__).resolve().parent
RUN_CKPTS = ROOT / "logs/train/2026-07-18_10-46-57_12345_local/checkpoints"
CHECKPOINTS = [
    (0, ROOT / "baseline/s_baseline.ckpt"),
    *[
        (step, RUN_CKPTS / f"step_{step:07d}.ckpt")
        for step in range(10_000, 100_000, 10_000)
    ],
]
N = 64
LENGTH = 256
VOCAB = 27
BLOCK_SIZE = 16
BATCH_SIZE = 32
SEED = 20260719


def stamp(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def exact_windows(path: Path, start: int, stop: int) -> torch.Tensor:
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    if not (0 <= start < stop <= len(raw)):
        raise ValueError(f"invalid region [{start}, {stop}) for {path} ({len(raw)} tokens)")
    max_start = stop - LENGTH
    starts = np.linspace(start, max_start, N).round().astype(np.int64)
    assert len(np.unique(starts)) == N
    assert np.diff(starts).min() >= LENGTH
    return torch.from_numpy(
        np.stack([raw[s : s + LENGTH] for s in starts]).astype(np.int64)
    )


def load_net_state(net: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    state = payload.get("state_dict", payload)
    state = {key.replace("net._orig_mod.", "net."): value for key, value in state.items()}
    net_state = {
        key.removeprefix("net."): value
        for key, value in state.items()
        if key.startswith("net.")
    }
    net.load_state_dict(net_state, strict=True)
    del net_state, state, payload


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    xt = x0 * (1.0 - t[..., None])
    xt.scatter_add_(-1, x1[..., None], t[..., None])
    return xt


@torch.inference_mode()
def evaluate_split(
    net,
    x1_cpu: torch.Tensor,
    x0_cpu: torch.Tensor,
    block_t_cpu: torch.Tensor,
    split: str,
) -> tuple[dict, dict]:
    all_ce, all_correct, all_conf, all_true_prob, all_pred_entropy = [], [], [], [], []
    all_times = []
    for lo in range(0, N, BATCH_SIZE):
        hi = min(N, lo + BATCH_SIZE)
        x1 = x1_cpu[lo:hi].cuda()
        x0 = x0_cpu[lo:hi].cuda()
        block_t = block_t_cpu[lo:hi].cuda()
        token_t = block_t.repeat_interleave(BLOCK_SIZE, dim=1)
        xt = interpolate(x0, x1, token_t)
        clean = F.one_hot(x1, VOCAB).to(x0.dtype)
        ones = torch.ones_like(token_t)
        doubled = torch.cat((clean, xt), dim=1)
        times = torch.cat((ones, token_t), dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(doubled, times, times)[:, LENGTH:]
        logits = logits.float()
        probs = logits.softmax(dim=-1)
        ce = F.cross_entropy(logits.transpose(1, 2), x1, reduction="none")
        prediction = logits.argmax(dim=-1)
        correct = prediction.eq(x1)
        confidence = probs.max(dim=-1).values
        true_prob = probs.gather(-1, x1[..., None]).squeeze(-1)
        pred_entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1)
        all_ce.append(ce.cpu())
        all_correct.append(correct.cpu())
        all_conf.append(confidence.cpu())
        all_true_prob.append(true_prob.cpu())
        all_pred_entropy.append(pred_entropy.cpu())
        all_times.append(token_t.cpu())

    ce = torch.cat(all_ce)
    correct = torch.cat(all_correct)
    confidence = torch.cat(all_conf)
    true_prob = torch.cat(all_true_prob)
    pred_entropy = torch.cat(all_pred_entropy)
    times = torch.cat(all_times)
    wrong = ~correct
    summary = {
        "split": split,
        "ce": float(ce.mean()),
        "accuracy": float(correct.float().mean()),
        "confidence": float(confidence.mean()),
        "wrong_token_confidence": float(confidence[wrong].mean()),
        "true_token_probability": float(true_prob.mean()),
        "predictive_entropy": float(pred_entropy.mean()),
        "n_sequences": N,
        "n_tokens": int(ce.numel()),
    }
    detail = {
        "split": split,
        "ce_by_block": [
            float(ce[:, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE].mean())
            for b in range(LENGTH // BLOCK_SIZE)
        ],
        "accuracy_by_block": [
            float(correct[:, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE].float().mean())
            for b in range(LENGTH // BLOCK_SIZE)
        ],
        "ce_by_time_bin": {},
        "accuracy_by_time_bin": {},
    }
    for bin_index, (left, right) in enumerate(
        zip((0.0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.0))
    ):
        mask = (times >= left) & (times < right)
        label = f"{left:.2f}-{right:.2f}"
        detail["ce_by_time_bin"][label] = float(ce[mask].mean())
        detail["accuracy_by_time_bin"][label] = float(
            correct[mask].float().mean()
        )
    return summary, detail


train_path = ROOT / "data/text8/train.bin"
validation_path = ROOT / "data/text8/val.bin"
train_tokens = train_path.stat().st_size // np.dtype(np.uint16).itemsize
train_seen = exact_windows(train_path, 0, 351_563)
train_unseen_tail = exact_windows(train_path, 351_563, train_tokens)
validation = exact_windows(validation_path, 0, 20_000)
generator = torch.Generator(device="cpu").manual_seed(SEED)
x0 = torch.randn((N, LENGTH, VOCAB), generator=generator)
block_t = torch.rand((N, LENGTH // BLOCK_SIZE), generator=generator)
torch.save(
    {
        "train_seen": train_seen,
        "train_unseen_tail": train_unseen_tail,
        "validation": validation,
        "x0": x0,
        "block_t": block_t,
        "seed": SEED,
    },
    OUTPUT / "fixed_probe_inputs.pt",
)

from block.block_dit import BlockDIT  # noqa: E402

net = BlockDIT(
    vocab_size=VOCAB,
    hidden_size=768,
    cond_dim=128,
    n_blocks=12,
    n_heads=12,
    dropout=0.1,
    length=LENGTH,
    block_size=BLOCK_SIZE,
    embed_type="naive",
    attention_backend="flex",
    jvp_attention_backend="auto",
    flex_kernel_block_size=64,
).cuda().eval()

rows, details = [], []
for step, checkpoint in CHECKPOINTS:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stamp(f"stage=checkpoint_load step={step} path={checkpoint}")
    load_net_state(net, checkpoint)
    for split, tokens in (
        ("train_seen", train_seen),
        ("train_unseen_tail", train_unseen_tail),
        ("validation", validation),
    ):
        started = time.perf_counter()
        summary, detail = evaluate_split(net, tokens, x0, block_t, split)
        summary["checkpoint_step"] = step
        summary["checkpoint"] = str(checkpoint)
        summary["seconds"] = time.perf_counter() - started
        detail["checkpoint_step"] = step
        rows.append(summary)
        details.append(detail)
        stamp(
            f"stage=probe step={step} split={split} ce={summary['ce']:.6f} "
            f"accuracy={summary['accuracy']:.4f} confidence={summary['confidence']:.4f} "
            f"wrong_confidence={summary['wrong_token_confidence']:.4f}"
        )

(OUTPUT / "fixed_probe.json").write_text(json.dumps(rows, indent=2))
(OUTPUT / "fixed_probe_details.json").write_text(json.dumps(details, indent=2))
with (OUTPUT / "fixed_probe.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by_step = {
    step: {
        row["split"]: row
        for row in rows
        if row["checkpoint_step"] == step
    }
    for step, _ in CHECKPOINTS
}
mechanism = []
for step, _ in CHECKPOINTS:
    train_row = by_step[step]["train_seen"]
    train_tail_row = by_step[step]["train_unseen_tail"]
    val_row = by_step[step]["validation"]
    mechanism.append(
        {
            "checkpoint_step": step,
            "train_seen_ce": train_row["ce"],
            "train_unseen_tail_ce": train_tail_row["ce"],
            "validation_ce": val_row["ce"],
            "generalization_gap_ce": val_row["ce"] - train_row["ce"],
            "train_seen_accuracy": train_row["accuracy"],
            "train_unseen_tail_accuracy": train_tail_row["accuracy"],
            "validation_accuracy": val_row["accuracy"],
            "train_seen_confidence": train_row["confidence"],
            "train_unseen_tail_confidence": train_tail_row["confidence"],
            "validation_confidence": val_row["confidence"],
            "validation_wrong_token_confidence": val_row["wrong_token_confidence"],
            "validation_predictive_entropy": val_row["predictive_entropy"],
        }
    )
(OUTPUT / "fixed_probe_mechanism.json").write_text(json.dumps(mechanism, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
steps = [row["checkpoint_step"] for row in mechanism]
axes[0].plot(
    steps,
    [row["train_seen_ce"] for row in mechanism],
    marker="o",
    label="seen train prefix",
)
axes[0].plot(
    steps,
    [row["train_unseen_tail_ce"] for row in mechanism],
    marker="o",
    label="unseen train tail",
)
axes[0].plot(
    steps, [row["validation_ce"] for row in mechanism], marker="o", label="validation"
)
axes[0].axhline(math.log(VOCAB), linestyle="--", color="#555555", label="ln(27)")
axes[0].set_title("Fixed-latent VFM cross-entropy")
axes[0].set_ylabel("CE")
axes[0].legend(fontsize=8)

axes[1].plot(
    steps,
    [row["train_seen_accuracy"] for row in mechanism],
    marker="o",
    label="seen train prefix",
)
axes[1].plot(
    steps,
    [row["train_unseen_tail_accuracy"] for row in mechanism],
    marker="o",
    label="unseen train tail",
)
axes[1].plot(
    steps,
    [row["validation_accuracy"] for row in mechanism],
    marker="o",
    label="validation",
)
axes[1].set_title("Token accuracy")
axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8)

axes[2].plot(
    steps,
    [row["validation_confidence"] for row in mechanism],
    marker="o",
    label="all val predictions",
)
axes[2].plot(
    steps,
    [row["validation_wrong_token_confidence"] for row in mechanism],
    marker="o",
    label="wrong val predictions",
)
axes[2].set_title("Validation confidence")
axes[2].set_ylabel("Maximum predicted probability")
axes[2].legend(fontsize=8)

for axis in axes:
    axis.set_xlabel("BCFM fine-tune step")
    axis.grid(alpha=0.25)
fig.suptitle("Shared data, Gaussian prior, and block times across every checkpoint")
fig.tight_layout()
fig.savefig(OUTPUT / "fixed_probe_dynamics.png", dpi=200, bbox_inches="tight")
plt.close(fig)

detail_by_key = {
    (item["checkpoint_step"], item["split"]): item for item in details
}
block_rows = []
for step in (0, 90_000):
    for split in ("train_seen", "train_unseen_tail", "validation"):
        for block_index, ce in enumerate(
            detail_by_key[(step, split)]["ce_by_block"], start=1
        ):
            block_rows.append(
                {
                    "checkpoint_step": step,
                    "split": split,
                    "block_index": block_index,
                    "ce": ce,
                }
            )
with (OUTPUT / "fixed_probe_by_block.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(block_rows[0]))
    writer.writeheader()
    writer.writerows(block_rows)

blocks = np.arange(1, LENGTH // BLOCK_SIZE + 1)
fig, axis = plt.subplots(figsize=(12, 6.4))
styles = (
    (0, "train_seen", "--", "#56B4E9", "Source CFM · seen train prefix"),
    (0, "validation", "--", "#009E73", "Source CFM · validation"),
    (90_000, "train_seen", "-", "#0072B2", "BCFM 90k · seen train prefix"),
    (
        90_000,
        "train_unseen_tail",
        "-",
        "#CC79A7",
        "BCFM 90k · unseen train tail",
    ),
    (90_000, "validation", "-", "#D55E00", "BCFM 90k · validation"),
)
for step, split, linestyle, color, label in styles:
    axis.plot(
        blocks,
        detail_by_key[(step, split)]["ce_by_block"],
        marker="o",
        linestyle=linestyle,
        color=color,
        label=label,
    )
axis.axhline(
    math.log(VOCAB),
    linestyle=":",
    color="#555555",
    label="uniform CE = ln(27)",
)
axis.set_title(
    "BCFM memorizes clean-prefix continuations after block 1", pad=14, fontsize=15
)
axis.set_xlabel("Block index (block size = 16 tokens)")
axis.set_ylabel("Fixed-latent VFM cross-entropy")
axis.set_xticks(blocks)
axis.grid(alpha=0.25)
axis.legend(fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT / "ce_by_block.png", dpi=200, bbox_inches="tight")
plt.close(fig)

stamp(f"stage=done output={OUTPUT / 'fixed_probe.csv'}")
