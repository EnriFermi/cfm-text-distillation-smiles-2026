#!/usr/bin/env python3
"""Compare full-sequence CFM and trained blockwise CFM sampling on one checkpoint.

The script deliberately keeps the SemiCat generative-PPL protocol:

* Text8 character sequences of length 256;
* synchronous copies of generated token IDs to CPU;
* GPT-2-large tokenizer and the requested causal-LM judge;
* token-weighted NLL followed by ``exp(NLL)``.

Unlike launching ``eval.run_eval`` once per point, the judge is loaded only once.
Every generated token tensor and every metric row is still written separately.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import transformers
from hydra import compose, initialize_config_dir
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))

from block.nfe import nfe_per_token, total_forwards  # noqa: E402
from block.sampling import block_causal_sample  # noqa: E402
from eval.generate import make_datamodule  # noqa: E402
from eval.metrics import entropy_per_block, entropy_per_block_per_sample  # noqa: E402
from semicat.metric.text_dist import TextMetrics  # noqa: E402


def _stamp(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_configs() -> tuple[DictConfig, DictConfig]:
    common = [
        "device=cuda",
        "model.net.embed_type=naive",
        "model.net.dropout=0.1",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        full = compose(
            config_name="eval_bcfm",
            overrides=["model=text8_dit_lg", "sampler=full_cfm", *common],
        )
        block = compose(
            config_name="eval_bcfm",
            overrides=["model=block_text8", "sampler=bcfm_train", *common],
        )
    return full, block


def _load_module(cfg: DictConfig, checkpoint: Path, device: str):
    """Strict-load on CPU with mmap, then move only model state to the GPU."""
    _stamp(f"stage=model_load class={cfg.model._target_} checkpoint={checkpoint}")
    module = hydra.utils.instantiate(cfg.model)
    payload = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    state = payload.get("state_dict", payload)
    state = {key.replace("net._orig_mod.", "net."): value for key, value in state.items()}
    module.load_state_dict(state, strict=True)
    del state, payload
    gc.collect()
    module = module.to(device).eval()
    _stamp(
        "stage=model_ready "
        f"class={type(module).__name__} device={device} "
        f"dtype={next(module.parameters()).dtype} "
        f"parameters={sum(p.numel() for p in module.parameters())}"
    )
    return module


@torch.inference_mode()
def _one_batch(module, formulation: str, nfe: int, batch_size: int, length: int):
    if formulation == "full_cfm":
        endpoint = module.sample_flow_map_batch(
            batch_size=batch_size, sampling_steps=nfe
        )
        return endpoint.argmax(dim=-1)
    if formulation == "blockwise_cfm":
        return block_causal_sample(
            module,
            block_size=module.block_size,
            steps_per_block=nfe,
            batch_size=batch_size,
            length=length,
            discretize="argmax",
        )
    raise ValueError(formulation)


@torch.inference_mode()
def _generate(
    module,
    formulation: str,
    nfe: int,
    n_samples: int,
    batch_size: int,
    length: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    warm_batch = min(8, batch_size, n_samples)
    _stamp(
        f"stage=warmup formulation={formulation} nfe={nfe} batch={warm_batch}"
    )
    _seed(seed)
    _one_batch(module, formulation, nfe, warm_batch, length)
    torch.cuda.synchronize()

    # The throwaway warm-up must not alter the evaluated random draws.
    _seed(seed)
    outputs: list[torch.Tensor] = []
    done = 0
    started = time.perf_counter()
    n_batches = math.ceil(n_samples / batch_size)
    for batch_index in range(n_batches):
        current_batch = min(batch_size, n_samples - done)
        tokens = _one_batch(
            module, formulation, nfe, current_batch, length
        )
        # Intentionally synchronous. This is the evaluation-side fix for the old
        # non_blocking CPU-transfer race.
        outputs.append(tokens.to("cpu", non_blocking=False))
        done += current_batch
        elapsed = time.perf_counter() - started
        _stamp(
            "stage=sampling "
            f"formulation={formulation} nfe={nfe} "
            f"batch={batch_index + 1}/{n_batches} samples={done}/{n_samples} "
            f"elapsed={elapsed:.1f}s samples_per_s={done / max(elapsed, 1e-9):.2f}"
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return torch.cat(outputs, dim=0)[:n_samples], elapsed


@torch.inference_mode()
def _judge_ppl(
    strings: list[str],
    tokenizer,
    judge,
    *,
    batch_size: int,
    context_size: int,
    label: str,
    device: str,
) -> tuple[float, float, int]:
    """Exact upstream token-weighted gen-PPL reduction, with progress logs."""
    samples, attention_mask = TextMetrics._retokenize(
        tokenizer, context_size, strings, device
    )
    batch_size = min(samples.size(0), batch_size)
    n_batches = samples.size(0) // batch_size
    if n_batches * batch_size != samples.size(0):
        raise ValueError(
            "n_samples must be divisible by judge_batch_size to preserve the exact "
            "upstream batching/reduction"
        )

    nll_sums: list[np.ndarray] = []
    effective_size = 0
    started = time.perf_counter()
    for batch_index in range(n_batches):
        sample_batch = samples[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        mask_batch = attention_mask[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        sample_chunks = torch.split(sample_batch, context_size, dim=-1)
        mask_chunks = torch.split(mask_batch, context_size, dim=-1)
        for sample_chunk, mask_chunk in zip(sample_chunks, mask_chunks):
            logits = judge(sample_chunk, attention_mask=mask_chunk).logits
            nlls = F.cross_entropy(
                logits.transpose(-1, -2)[..., :-1],
                sample_chunk[..., 1:],
                reduction="none",
            )
            first_eos = (
                sample_chunk == tokenizer.eos_token_id
            ).cumsum(-1) == 1
            token_mask = sample_chunk != tokenizer.eos_token_id
            valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
            nll_sums.append((nlls * valid_tokens).sum().cpu().numpy())
            effective_size += valid_tokens.sum().item()

        if (
            batch_index == 0
            or (batch_index + 1) % 32 == 0
            or batch_index + 1 == n_batches
        ):
            elapsed = time.perf_counter() - started
            _stamp(
                "stage=judge "
                f"label={label} batch={batch_index + 1}/{n_batches} "
                f"elapsed={elapsed:.1f}s batches_per_s="
                f"{(batch_index + 1) / max(elapsed, 1e-9):.2f}"
            )

    mean_nll = float(sum(nll_sums) / effective_size)
    return float(np.exp(mean_nll)), mean_nll, int(effective_size)


def _write_summary(rows: list[dict], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--formulations",
        nargs="+",
        choices=["full_cfm", "blockwise_cfm"],
        default=["full_cfm", "blockwise_cfm"],
    )
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--judge-model", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    full_cfg, block_cfg = _compose_configs()
    metadata_payload = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    checkpoint_step = int(metadata_payload.get("global_step", -1))
    checkpoint_epoch = int(metadata_payload.get("epoch", -1))
    del metadata_payload

    protocol = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_epoch": checkpoint_epoch,
        "formulations": args.formulations,
        "nfe": args.nfe,
        "n_samples": args.n_samples,
        "sample_batch_size": args.sample_batch_size,
        "judge_batch_size": args.judge_batch_size,
        "judge_model": args.judge_model,
        "judge_tokenizer": "gpt2-large",
        "length": args.length,
        "block_size": int(block_cfg.model.block_size),
        "report_block_size": int(block_cfg.model.block_size),
        "seed": args.seed,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else None
        ),
        "sampling_parameter_dtype": "torch.float32",
        "discretize": "argmax",
        "cpu_transfer_non_blocking": False,
        "git_commit": _git_commit(),
        "full_model_config": OmegaConf.to_container(full_cfg.model, resolve=True),
        "block_model_config": OmegaConf.to_container(block_cfg.model, resolve=True),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    _stamp(
        "stage=start "
        f"checkpoint_step={checkpoint_step} checkpoint={checkpoint} "
        f"device={args.device} gpu={protocol['gpu']} "
        f"n_samples={args.n_samples} seed={args.seed} nfe={args.nfe} "
        f"output={output_dir}"
    )

    # Text8 setup is kept on CPU: only its exact upstream itos mapping is needed.
    _stamp("stage=data_setup device=cpu dataset=text8")
    datamodule = make_datamodule(full_cfg, "cpu")

    generated: list[dict] = []
    rows: list[dict] = []
    report_block_size = int(block_cfg.model.block_size)
    config_by_formulation = {
        "full_cfm": full_cfg,
        "blockwise_cfm": block_cfg,
    }
    for formulation in args.formulations:
        module = _load_module(
            config_by_formulation[formulation], checkpoint, args.device
        )
        block_size = (
            args.length if formulation == "full_cfm" else int(module.block_size)
        )
        for nfe in args.nfe:
            tokens, sampling_seconds = _generate(
                module,
                formulation,
                nfe,
                args.n_samples,
                args.sample_batch_size,
                args.length,
                args.seed,
            )
            label = f"{formulation}_nfe{nfe}"
            strings = datamodule.tensor_to_strings(tokens)
            torch.save(tokens, output_dir / f"{label}_tokens.pt")
            (output_dir / f"{label}_samples.txt").write_text("\n".join(strings))
            pooled_report_blocks = entropy_per_block(tokens, report_block_size)
            per_sample_report_blocks = entropy_per_block_per_sample(
                tokens, report_block_size
            )

            row = {
                "formulation": formulation,
                "nfe": int(nfe),
                "nfe_semantics": (
                    "full_sequence_steps"
                    if formulation == "full_cfm"
                    else "steps_per_block"
                ),
                "block_size": block_size,
                "total_model_forwards_per_sequence": total_forwards(
                    block_size, nfe, args.length
                ),
                "nfe_per_token": nfe_per_token(block_size, nfe, args.length),
                "checkpoint_step": checkpoint_step,
                "n_samples": args.n_samples,
                "length": args.length,
                "seed": args.seed,
                "sampling_seconds": sampling_seconds,
                "sampling_tokens_per_second": (
                    args.n_samples * args.length / sampling_seconds
                ),
                "pooled_entropy_nats": entropy_per_block(
                    tokens, args.length
                )[0],
                "mean_per_sample_generation_block_entropy_nats": float(
                    np.mean(
                        entropy_per_block_per_sample(tokens, block_size)
                    )
                ),
                "report_block_size": report_block_size,
                "pooled_report_block_entropy_mean_nats": float(
                    np.mean(pooled_report_blocks)
                ),
                "mean_per_sample_report_block_entropy_nats": float(
                    np.mean(per_sample_report_blocks)
                ),
                "mean_nll": None,
                "gen_ppl": None,
                "effective_judge_tokens": None,
            }
            generated.append({"label": label, "strings": strings, "row": row})
            rows.append(row)
            _write_summary(rows, output_dir)
            _stamp(
                "stage=samples_written "
                f"label={label} tokens={output_dir / f'{label}_tokens.pt'} "
                f"text={output_dir / f'{label}_samples.txt'}"
            )

        del module
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _stamp(f"stage=model_released formulation={formulation}")

    if not args.skip_judge:
        _stamp(
            "stage=judge_load "
            f"model={args.judge_model} tokenizer=gpt2-large device={args.device}"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = TextMetrics._load_tokenizer()
        judge = transformers.AutoModelForCausalLM.from_pretrained(
            args.judge_model
        ).eval().to(args.device)
        _stamp(
            "stage=judge_ready "
            f"model={args.judge_model} tokenizer={tokenizer.name_or_path} "
            f"dtype={next(judge.parameters()).dtype}"
        )
        for item in generated:
            ppl, mean_nll, effective_tokens = _judge_ppl(
                item["strings"],
                tokenizer,
                judge,
                batch_size=args.judge_batch_size,
                context_size=args.length,
                label=item["label"],
                device=args.device,
            )
            item["row"]["mean_nll"] = mean_nll
            item["row"]["gen_ppl"] = ppl
            item["row"]["effective_judge_tokens"] = effective_tokens
            _write_summary(rows, output_dir)
            _stamp(
                "stage=metric "
                f"label={item['label']} mean_nll={mean_nll:.6f} "
                f"gen_ppl={ppl:.6f} effective_tokens={effective_tokens}"
            )
        del judge
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    _write_summary(rows, output_dir)
    _stamp(
        f"stage=done summary_json={output_dir / 'summary.json'} "
        f"summary_csv={output_dir / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
