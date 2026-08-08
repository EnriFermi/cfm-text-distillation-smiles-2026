#!/usr/bin/env python3
"""Join matched M1/MDLM BF16 quality and latency, then build the all-model table."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd


def _quality_model(label: str) -> str:
    if "M1_200k" in label:
        return "M1-200k"
    if "M1_68k" in label:
        return "M1-68k"
    if "MDLM" in label:
        return "MDLM"
    raise ValueError(f"cannot infer model from quality label {label!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path, required=True)
    parser.add_argument("--existing-optimized-csv", type=Path, required=True)
    parser.add_argument("--previous-quality-latency", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[config] "
        + json.dumps(
            {
                "latency_json": str(args.latency_json.resolve()),
                "quality_json": str(args.quality_json.resolve()),
                "existing_optimized_csv": str(args.existing_optimized_csv.resolve()),
                "previous_quality_latency": (
                    str(args.previous_quality_latency.resolve())
                    if args.previous_quality_latency else None
                ),
                "output_dir": str(output_dir),
                "device": "stored measurements: NVIDIA H100 NVL",
                "dtype": "BF16 generation/latency; FP32 GPT-J judge",
                "seed": 0,
                "cache_mode": "stored matched artifacts",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    print("[stage=load] reading latency, quality, and existing optimized rows", flush=True)
    latency_payload = json.loads(args.latency_json.read_text())
    quality_payload = json.loads(args.quality_json.read_text())
    latency = pd.DataFrame(latency_payload["rows"])
    quality = pd.DataFrame(quality_payload["rows"])
    quality["model"] = quality.label.map(_quality_model)
    expected_models = {"M1-68k", "M1-200k", "MDLM"}
    if set(latency.model) != expected_models or set(quality.model) != expected_models:
        raise RuntimeError(
            f"model mismatch: latency={sorted(set(latency.model))}, "
            f"quality={sorted(set(quality.model))}"
        )
    if len(latency) != 17 or len(quality) != 17:
        raise RuntimeError(f"expected 17+17 rows, got {len(latency)}+{len(quality)}")
    if latency.duplicated(["model", "point"]).any():
        raise RuntimeError("duplicate latency (model, point)")
    if quality.duplicated(["model", "point"]).any():
        raise RuntimeError("duplicate quality (model, point)")

    print("[stage=join] exact one-to-one join on model and operating point", flush=True)
    quality_columns = quality[
        ["model", "point", "mauve", "gen_ppl", "n_samples", "dump", "label"]
    ].rename(
        columns={
            "n_samples": "quality_n_samples",
            "dump": "quality_dump",
            "label": "quality_label",
        }
    )
    joined = latency.merge(
        quality_columns,
        on=["model", "point"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 17:
        raise RuntimeError(f"join produced {len(joined)} rows instead of 17")
    joined["steps_per_block"] = joined["nfe"].astype(int)
    joined["total_denoising_nfe"] = joined["nfe"].astype(int)
    joined["separate_cache_calls"] = 0
    joined["dtype"] = joined["precision"]
    joined["quality_source"] = str(args.quality_json.resolve())
    joined["latency_source"] = str(args.latency_json.resolve())

    required_numeric = [
        "mauve",
        "gen_ppl",
        "sequence_latency_ms",
        "sequence_latency_p10_ms",
        "sequence_latency_p90_ms",
    ]
    if not joined[required_numeric].map(math.isfinite).all().all():
        raise RuntimeError("joined metrics contain NaN/inf")
    if not (joined.quality_n_samples == 512).all():
        raise RuntimeError("quality sample count is not uniformly 512")
    if not (joined.sequence_latency_repeats == 30).all():
        raise RuntimeError("latency repeat count is not uniformly 30")
    if set(joined.precision) != {"bf16"} or set(joined.compile_mode) != {"max-autotune"}:
        raise RuntimeError("latency precision/compile policy is not BF16/max-autotune")

    joined = joined.sort_values(["model", "nfe"]).reset_index(drop=True)
    joined_path = output_dir / "m1_mdlm_optimized_quality_latency.csv"
    joined.to_csv(joined_path, index=False)

    delta_path = None
    if args.previous_quality_latency:
        names = {
            "CFM M1 (100k target; best@68k)": "M1-68k",
            "CFM M1 (200k)": "M1-200k",
            "MDLM": "MDLM",
        }
        previous = pd.read_csv(args.previous_quality_latency)
        previous = previous[previous.model.isin(names)].copy()
        previous["model"] = previous.model.map(names)
        previous = previous[
            ["model", "point", "latency_ms", "mauve", "gen_ppl"]
        ].rename(
            columns={
                "latency_ms": "previous_latency_ms",
                "mauve": "previous_mauve",
                "gen_ppl": "previous_gen_ppl",
            }
        )
        deltas = joined.merge(previous, on=["model", "point"], validate="one_to_one")
        deltas["latency_speedup"] = deltas.previous_latency_ms / deltas.sequence_latency_ms
        deltas["mauve_delta"] = deltas.mauve - deltas.previous_mauve
        deltas["gen_ppl_delta"] = deltas.gen_ppl - deltas.previous_gen_ppl
        delta_path = output_dir / "delta_vs_previous_fp32.csv"
        deltas.sort_values(["model", "nfe"]).to_csv(delta_path, index=False)

    previous_optimized = pd.read_csv(args.existing_optimized_csv)
    if set(previous_optimized.model) != {"M2", "M3", "BD3-LM"} or len(previous_optimized) != 19:
        raise RuntimeError("existing optimized table is not the expected 19-row M2/M3/BD3 set")
    all_models = pd.concat([previous_optimized, joined], ignore_index=True, sort=False)
    if len(all_models) != 36 or set(all_models.model) != {
        "M1-68k", "M1-200k", "M2", "M3", "BD3-LM", "MDLM",
    }:
        raise RuntimeError("combined optimized table does not contain the expected 36 points")
    if all_models.duplicated(["model", "point"]).any():
        raise RuntimeError("combined table has duplicate operating points")
    if set(all_models.dtype) != {"bf16"} or set(all_models.compile_mode) != {"max-autotune"}:
        raise RuntimeError("combined table is not uniformly BF16/max-autotune")
    all_models = all_models.sort_values(["model", "steps_per_block", "point"]).reset_index(drop=True)
    all_models_path = output_dir / "all_models_optimized_quality_latency.csv"
    all_models.to_csv(all_models_path, index=False)

    metadata = {
        "schema": "bcfm-all-models-bf16-quality-latency/v1",
        "rows": len(all_models),
        "models": sorted(all_models.model.unique()),
        "quality_protocol": {
            "samples_per_point": 512,
            "mauve_featurizer": "gpt2-large",
            "mauve_context": 256,
            "judge": quality_payload.get("judge"),
            "judge_dtype": "float32",
        },
        "latency_protocol": latency_payload["metadata"],
        "artifacts": {
            "m1_mdlm_csv": str(joined_path),
            "all_models_csv": str(all_models_path),
            "delta_csv": str(delta_path) if delta_path else None,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    report_lines = [
        "# Matched BF16/max-autotune M1 and MDLM results",
        "",
        "Every listed point uses H100 NVL batch-one latency (5 warmups, median of "
        "30 repeats) and newly generated BF16 quality samples (n=512).",
        "",
        "| model | point | latency ms | MAUVE | gen-PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in joined.itertuples():
        report_lines.append(
            f"| {row.model} | {row.point} | {row.sequence_latency_ms:.3f} | "
            f"{row.mauve:.6f} | {row.gen_ppl:.3f} |"
        )
    report_lines.extend(
        [
            "",
            f"M1/MDLM data: `{joined_path}`",
            f"All-model data: `{all_models_path}`",
            f"Metadata: `{metadata_path}`",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(
        f"[done] rows={len(all_models)} elapsed={time.perf_counter() - started:.1f}s "
        f"joined={joined_path} all_models={all_models_path} deltas={delta_path} "
        f"metadata={metadata_path} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
