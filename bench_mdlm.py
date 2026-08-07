"""Generation latency of the TinyStories MDLM checkpoint.

For an equal total denoiser budget against 16-block CFM/BD3-LM, compare MDLM
``num_steps = 16 * steps_per_block``.  The default sweep therefore uses
16/32/64/128/256 full-sequence diffusion steps.

    python "eval test/bench_mdlm.py" --fast --compile --amp bf16
"""

from __future__ import annotations

import argparse
import contextlib

import torch

from common import (
    MDLM_CKPT,
    bootstrap,
    gpu_name,
    peak_mem_gb,
    row,
    save,
    timed,
)

bootstrap("mdlm")

import mdlm_fast  # noqa: E402
from bcfm_baselines.checkpoint import load_checkpoint_model  # noqa: E402
from bcfm_baselines.models.generative_text import GenerationConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", nargs="+", type=int,
                        default=[16, 32, 64, 128, 256])
    parser.add_argument("--batch-size", nargs="+", type=int,
                        default=[1, 4, 16, 32])
    parser.add_argument("--fast", action="store_true",
                        help="project/sample masked positions only (same distribution, "
                             "different seed mapping)")
    parser.add_argument("--lean", action="store_true",
                        help="remove diagnostics but preserve seed-for-seed tokens")
    parser.add_argument("--amp", choices=["bf16", "fp16"], default=None)
    parser.add_argument("--compile", action="store_true",
                        help="compile the static full-sequence Transformer")
    parser.add_argument("--compile-mode",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        default="default")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=str(MDLM_CKPT))
    parser.add_argument("--out", default="mdlm_latency.json")
    args = parser.parse_args()
    if args.fast and args.lean:
        parser.error("--fast already includes the lean path; choose only one")
    if args.compile_mode != "default" and not args.compile:
        parser.error("--compile-mode requires --compile")

    if args.amp is None:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model, payload = load_checkpoint_model(args.checkpoint, args.device)
    if model.model_type != "mdlm":
        raise SystemExit(f"checkpoint model is {model.model_type!r}, expected 'mdlm'")
    config = payload["config"]
    base_sampling = dict(config["sampling"])
    length = base_sampling.get("length", 256)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)

    body = None
    if args.fast:
        body = mdlm_fast.make_hidden_body(
            model,
            compile_body=args.compile,
            compile_mode=args.compile_mode,
        )
    elif args.compile:
        compile_kwargs = (
            {} if args.compile_mode == "default"
            else {"mode": args.compile_mode}
        )
        model.backbone = torch.compile(model.backbone, **compile_kwargs)

    mode = "active-fast" if args.fast else "lean" if args.lean else "reference"
    opts = [mode]
    if args.amp:
        opts.append(args.amp)
    if args.compile:
        opts.append(f"torch.compile({args.compile_mode})")
    print(
        f"[mdlm] step={payload.get('step', payload.get('global_step'))} "
        f"weights={payload['weights_used_for_inference']} params={n_params:,} "
        f"length={length} hidden={model.backbone.config.hidden_size}x"
        f"{model.backbone.config.num_layers}"
    )
    print(f"[opt] {', '.join(opts)}")
    print(f"[gpu] {gpu_name()}\n")

    rows: list[dict] = []
    for batch_size in args.batch_size:
        for num_steps in args.num_steps:
            generation = GenerationConfig(**{
                **base_sampling,
                "batch_size": batch_size,
                "length": length,
                "num_steps": num_steps,
            })
            torch.cuda.reset_peak_memory_stats()
            result_holder = {}

            def once(generation=generation):
                if args.fast:
                    result = mdlm_fast.fast_generate(
                        model, generation, body=body, amp_dtype=amp_dtype
                    )
                elif args.lean:
                    result = mdlm_fast.lean_generate(
                        model, generation, amp_dtype=amp_dtype
                    )
                else:
                    amp = (
                        torch.autocast("cuda", dtype=amp_dtype)
                        if amp_dtype is not None
                        else contextlib.nullcontext()
                    )
                    with torch.inference_mode(), amp:
                        result = model.generate(generation)
                result_holder["result"] = result

            try:
                stats = timed(once, warmup=args.warmup, repeats=args.repeats)
            except torch.cuda.OutOfMemoryError:
                print(f"  batch={batch_size:3d} steps={num_steps:3d}  OOM")
                torch.cuda.empty_cache()
                continue

            actual_nfe = result_holder["result"].diagnostics[
                "num_function_evaluations"
            ]
            entry = row(
                model="mdlm",
                batch_size=batch_size,
                length=length,
                stats=stats,
                sampler=mode,
                num_steps=num_steps,
                nfe_per_sequence=actual_nfe,
                fast=args.fast,
                lean=args.lean or args.fast,
                active_position_rng=args.fast,
                amp=args.amp,
                compiled=args.compile,
                compile_mode=args.compile_mode if args.compile else None,
                peak_mem_gb=round(peak_mem_gb(), 2),
            )
            rows.append(entry)
            print(
                f"  batch={batch_size:3d} steps={num_steps:3d}  "
                f"{entry['latency_s_per_batch']:8.4f} s/batch  "
                f"{entry['latency_ms_per_sequence']:9.2f} ms/seq  "
                f"{entry['tokens_per_sec']:10.1f} tok/s  "
                f"NFE={actual_nfe:3d}  peak {entry['peak_mem_gb']:.2f} GB"
            )
            torch.cuda.empty_cache()

    save(rows, args.out, {
        "model": "mdlm",
        "checkpoint": args.checkpoint,
        "weights_used_for_inference": payload["weights_used_for_inference"],
        "checkpoint_step": payload.get("step", payload.get("global_step")),
        "n_params": n_params,
        "length": length,
        "gpu": gpu_name(),
        "device": args.device,
        "torch": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": args.amp or "fp32",
        "mode": mode,
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "base_sampling": base_sampling,
        "tf32": False if args.amp is None else None,
    })


if __name__ == "__main__":
    main()
