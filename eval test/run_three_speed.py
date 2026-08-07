"""Run the reproducible maximum-speed sweep for all three checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def run(arguments: list[str]) -> None:
    command = [sys.executable, *arguments]
    print("\n[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", nargs="+", type=int,
                        default=[1, 4, 16, 32])
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    common = [
        "--batch-size", *map(str, args.batch_size),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
    ]

    # On this RTX 5070 Laptop, the FP32 cached CFM path is faster than BF16;
    # compile applies only to its uncached fixed-shape path and loses at all but
    # the smallest point.  These are deliberately fastest measured profiles,
    # not a mechanical list of flags.
    run([
        "bench_cfm.py", "--nfe", *map(str, args.steps), *common,
        "--fast", "--cache", "--out", "cfm_latency_three_max.json",
    ])
    run([
        "bench_bd3lm.py", "--steps-per-block", *map(str, args.steps), *common,
        "--lean", "--compile", "--amp", "bf16",
        "--out", "bd3lm_latency_three_max.json",
    ])
    mdlm_steps = [16 * step for step in args.steps]
    run([
        "bench_mdlm.py", "--num-steps", *map(str, mdlm_steps), *common,
        "--fast", "--compile", "--amp", "bf16",
        "--out", "mdlm_latency_three_max.json",
    ])
    run(["analyze_three.py"])


if __name__ == "__main__":
    main()
