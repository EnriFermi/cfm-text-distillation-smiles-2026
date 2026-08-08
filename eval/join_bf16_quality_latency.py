#!/usr/bin/env python3
"""Join independently rescored BF16 samples with matched H100 latency rows."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd


def _quality_model(label: str) -> str:
    if "M2" in label:
        return "M2"
    if "M3" in label:
        return "M3"
    if "BD3" in label:
        return "BD3-LM"
    raise ValueError(f"cannot infer model from quality label {label!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path, required=True)
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
                "previous_quality_latency": (
                    str(args.previous_quality_latency.resolve())
                    if args.previous_quality_latency else None
                ),
                "output_dir": str(output_dir),
                "device": "stored measurements: NVIDIA H100 NVL",
                "dtype": "BF16 generation; FP32 GPT-J judge",
                "seed": 0,
                "cache_mode": "stored latency and independently rescored dumps",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    print("[stage=load] reading latency and quality artifacts", flush=True)
    latency_payload = json.loads(args.latency_json.read_text())
    quality_payload = json.loads(args.quality_json.read_text())
    latency = pd.DataFrame(latency_payload["rows"])
    latency = latency[
        (latency.precision == "bf16")
        & (latency.compile_mode == "max-autotune")
        & ((latency.model == "BD3-LM") | (latency.discretize == "argmax"))
    ].copy()
    quality = pd.DataFrame(quality_payload["rows"])
    quality["model"] = quality.label.map(_quality_model)

    expected_models = {"M2", "M3", "BD3-LM"}
    if set(latency.model) != expected_models or set(quality.model) != expected_models:
        raise RuntimeError(
            f"model mismatch: latency={sorted(set(latency.model))}, "
            f"quality={sorted(set(quality.model))}"
        )
    if len(latency) != 19 or len(quality) != 19:
        raise RuntimeError(f"expected 19+19 rows, got {len(latency)}+{len(quality)}")
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
    if len(joined) != 19:
        missing_latency = sorted(
            set(zip(latency.model, latency.point))
            - set(zip(joined.model, joined.point))
        )
        missing_quality = sorted(
            set(zip(quality.model, quality.point))
            - set(zip(joined.model, joined.point))
        )
        raise RuntimeError(
            f"join produced {len(joined)} rows; missing latency={missing_latency}; "
            f"missing quality={missing_quality}"
        )
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

    deltas = None
    if args.previous_quality_latency:
        previous = pd.read_csv(args.previous_quality_latency)
        previous = previous[["model", "point", "mauve", "gen_ppl"]].rename(
            columns={"mauve": "previous_mauve", "gen_ppl": "previous_gen_ppl"}
        )
        deltas = joined.merge(previous, on=["model", "point"], validate="one_to_one")
        deltas["mauve_delta"] = deltas.mauve - deltas.previous_mauve
        deltas["gen_ppl_delta"] = deltas.gen_ppl - deltas.previous_gen_ppl

    joined = joined.sort_values(["model", "steps_per_block", "point"]).reset_index(drop=True)
    csv_path = output_dir / "optimized_quality_latency.csv"
    joined.to_csv(csv_path, index=False)

    delta_path = None
    if deltas is not None:
        delta_path = output_dir / "quality_delta_vs_previous_fp32.csv"
        deltas.sort_values(["model", "steps_per_block", "point"]).to_csv(delta_path, index=False)

    metadata = {
        "schema": "bcfm-bf16-quality-latency/v1",
        "rows": len(joined),
        "models": sorted(joined.model.unique()),
        "quality_protocol": {
            "samples_per_point": 512,
            "mauve_featurizer": "gpt2-large",
            "mauve_context": 256,
            "judge": quality_payload.get("judge"),
            "judge_dtype": "float32",
        },
        "latency_protocol": latency_payload["metadata"],
        "artifacts": {
            "joined_csv": str(csv_path),
            "quality_delta_csv": str(delta_path) if delta_path else None,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    report_lines = [
        "# BF16 max-autotune quality/latency join",
        "",
        "All M2, M3, and BD3-LM rows use newly generated BF16 samples and matched "
        "BF16/max-autotune latency on the same H100 NVL.",
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
            f"Joined data: `{csv_path}`",
            f"Metadata: `{metadata_path}`",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(
        f"[done] rows={len(joined)} elapsed={time.perf_counter() - started:.1f}s "
        f"joined={csv_path} deltas={delta_path} metadata={metadata_path} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
