"""Join fused-sampler latency and quality evidence into reviewed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


COLORS = {
    "M2": "#1D7874",
    "M3": "#E76F51",
    "BD3-LM": "#5E4FA2",
}
MARKERS = {"M2": "o", "M3": "s", "BD3-LM": "D"}


def _json_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def _load_bd3_quality(path: Path) -> dict[str, dict]:
    columns = [
        "canonical_for_plot", "model_family", "model_variant",
        "evaluation_label", "point", "n_samples", "metric_name", "metric_value",
        "source_file",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[
        (frame.model_family == "BD3-LM")
        & (frame.model_variant == "best_canonical_100k")
        & frame.canonical_for_plot.astype(bool)
        & frame.metric_name.isin(["mauve", "gen_ppl"])
        & frame.evaluation_label.isin(["BD3LM_best_sweep", "BD3LM_best_native"])
    ].copy()
    wanted_label = frame.point.map(
        lambda point: "BD3LM_best_native" if point == "sample_nfe256" else "BD3LM_best_sweep"
    )
    frame = frame[frame.evaluation_label == wanted_label]
    if frame.duplicated(["point", "metric_name"]).any():
        raise RuntimeError("ambiguous canonical BD3-LM quality rows")
    values = frame.pivot(index="point", columns="metric_name", values="metric_value")
    counts = frame.groupby("point").n_samples.first()
    sources = frame.groupby("point").source_file.first()
    return {
        point: {
            "mauve": float(row.mauve),
            "gen_ppl": float(row.gen_ppl),
            "quality_n_samples": int(counts.loc[point]),
            "quality_source": str(sources.loc[point]),
        }
        for point, row in values.iterrows()
    }


def _style_axes(axis) -> None:
    axis.set_facecolor("#FBFAF7")
    axis.grid(True, which="major", color="#D9D5CC", linewidth=0.8, alpha=0.75)
    axis.grid(True, which="minor", color="#E9E6DF", linewidth=0.5, alpha=0.55)
    axis.spines[["top", "right"]].set_visible(False)


def _markdown_latency_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| steps/block | " + " | ".join(columns) + " |"
    divider = "|---:|" + "|".join("---:" for _ in columns) + "|"
    rows = [
        "| " + str(index) + " | "
        + " | ".join(f"{float(value):.3f}" for value in row)
        + " |"
        for index, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def _quality_plot(frame: pd.DataFrame, metric: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.4, 6.1), constrained_layout=True)
    fig.patch.set_facecolor("#F6F3EC")
    _style_axes(axis)
    for model in ("M2", "M3", "BD3-LM"):
        subset = frame[(frame.model == model) & ~frame.sampler.str.contains("first_hitting")]
        subset = subset.sort_values("total_denoising_nfe")
        axis.plot(
            subset.sequence_latency_ms,
            subset[metric],
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=7,
            linewidth=2.3,
            label=model,
        )
        for index, row in subset.reset_index(drop=True).iterrows():
            y_offset = {"M2": 9, "M3": -15}.get(
                model, 8 if index % 2 == 0 else -14,
            )
            axis.annotate(
                f"{int(row.steps_per_block)}",
                (row.sequence_latency_ms, row[metric]),
                xytext=(6, y_offset),
                textcoords="offset points",
                fontsize=8.5,
                color=COLORS[model],
            )
    first_hitting = frame[frame.sampler.str.contains("first_hitting")]
    if not first_hitting.empty:
        row = first_hitting.iloc[0]
        axis.scatter(
            row.sequence_latency_ms, row[metric], s=145, marker="*",
            color="#C43C55", edgecolor="white", linewidth=0.8,
            label="BD3-LM first-hitting",
            zorder=5,
        )
        axis.annotate(
            "FH", (row.sequence_latency_ms, row[metric]),
            xytext=(8, -17 if metric == "mauve" else 8),
            textcoords="offset points", fontsize=9, color="#9B1D3D",
        )
    axis.set_xscale("log")
    if metric == "gen_ppl":
        axis.set_yscale("log")
    axis.set_xlabel("Sequence latency on H100 NVL (ms; batch=1, length=256)")
    axis.set_ylabel("MAUVE ↑" if metric == "mauve" else "Generative perplexity ↓")
    axis.set_title(
        "TinyStories: quality vs optimized sequence latency",
        fontsize=15, weight="semibold", pad=12,
    )
    axis.legend(frameon=False, ncol=2, loc="best")
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _speedup_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    fig.patch.set_facecolor("#F6F3EC")
    _style_axes(axis)
    selected = frame[
        ((frame.model.isin(["M2", "M3"])) & (frame.discretize == "argmax"))
        | ((frame.model == "BD3-LM") & ~frame.sampler.str.contains("first_hitting"))
    ]
    for model in ("M2", "M3", "BD3-LM"):
        subset = selected[selected.model == model].sort_values("steps_per_block")
        axis.plot(
            subset.steps_per_block, subset.speedup_vs_previous_fast,
            color=COLORS[model], marker=MARKERS[model], markersize=7,
            linewidth=3.6 if model == "M2" else 2.1,
            linestyle="--" if model == "M2" else "-",
            zorder=2 if model == "M2" else 3,
            label=model,
        )
    axis.axhline(1.0, color="#5E5B55", linewidth=1.0, linestyle="--")
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    axis.set_xlabel("Denoising steps per block")
    axis.set_ylabel("Speedup over previous H100 fast path (×) ↑")
    axis.set_title("TinyStories: fused-cache sampler speedup", fontsize=15, weight="semibold", pad=12)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latency", type=Path,
        default=ROOT / "artifacts/tinystories_fused_sampler_h100_20260807/fused_sampler_latency.json",
    )
    parser.add_argument(
        "--new-quality", type=Path,
        default=ROOT / "results/ts_M2_M3_fused_fast_argmax_scores.json",
    )
    parser.add_argument(
        "--measurement-long-csv", type=Path,
        default=ROOT.parents[1] / "tinystories_all_measurements_long.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/tinystories_fused_sampler_h100_20260807",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[config] latency={args.latency.resolve()} quality={args.new_quality.resolve()} "
        f"measurement_long={args.measurement_long_csv.resolve()} "
        f"output={args.output_dir.resolve()}", flush=True,
    )

    latency = pd.DataFrame(_json_rows(args.latency))
    new_quality_rows = _json_rows(args.new_quality)
    new_quality = {(row["label"], row["point"]): row for row in new_quality_rows}
    old_quality: dict[tuple[str, str], dict] = {}
    old_paths = {
        "M2": ROOT / "results/ts_M2_source_fast_cached_argmax_scores.json",
        "M3": ROOT / "results/ts_M3_step100000_fast_cached_scores.json",
    }
    for model, path in old_paths.items():
        old_quality.update({(model, row["point"]): row for row in _json_rows(path)})
    bd3_quality = _load_bd3_quality(args.measurement_long_csv)
    ablation_path = args.output_dir / "cache_backend_ablation.json"
    parity_path = args.output_dir / "bd3_token_parity.json"
    ablation = json.loads(ablation_path.read_text())
    parity = json.loads(parity_path.read_text())
    if not all(row["exact_token_equal"] for row in parity["rows"]):
        raise RuntimeError("BD3-LM token parity artifact contains a failure")

    joined: list[dict] = []
    for row in latency.to_dict("records"):
        model = row["model"]
        if model in {"M2", "M3"}:
            if row["discretize"] != "argmax":
                continue
            label = "ts_M2_source_fused_fast" if model == "M2" else "ts_M3_step100000_fused_fast"
            quality = new_quality[(label, row["point"])]
            provenance = str(args.new_quality)
        else:
            quality = bd3_quality[row["point"]]
            provenance = quality["quality_source"]
        joined.append({
            **row,
            "mauve": quality["mauve"],
            "gen_ppl": quality["gen_ppl"],
            "quality_n_samples": quality.get("n_samples", quality.get("quality_n_samples")),
            "quality_source": provenance,
        })
    curves = pd.DataFrame(joined).sort_values(["model", "sampler", "total_denoising_nfe"])
    curves_path = args.output_dir / "optimized_quality_latency.csv"
    curves.to_csv(curves_path, index=False)

    deltas: list[dict] = []
    for row in curves[curves.model.isin(["M2", "M3"])].to_dict("records"):
        old = old_quality[(row["model"], row["point"])]
        deltas.append({
            "model": row["model"], "point": row["point"],
            "steps_per_block": row["steps_per_block"],
            "old_mauve": old["mauve"], "new_mauve": row["mauve"],
            "delta_mauve": row["mauve"] - old["mauve"],
            "old_gen_ppl": old["gen_ppl"], "new_gen_ppl": row["gen_ppl"],
            "delta_gen_ppl": row["gen_ppl"] - old["gen_ppl"],
        })
    deltas_frame = pd.DataFrame(deltas)
    deltas_path = args.output_dir / "m2_m3_quality_deltas.csv"
    deltas_frame.to_csv(deltas_path, index=False)

    drift_rows: list[dict] = []
    token_paths = {
        "M2": (
            ROOT / "dumps/ts_M2_source_fast_cached_argmax.tokens.npz",
            ROOT / "dumps/ts_M2_source_fused_fast.tokens.npz",
        ),
        "M3": (
            ROOT / "dumps/ts_M3_step100000_fast_cached.tokens.npz",
            ROOT / "dumps/ts_M3_step100000_fused_fast.tokens.npz",
        ),
    }
    for model, (old_path, new_path) in token_paths.items():
        old_tokens, new_tokens = np.load(old_path), np.load(new_path)
        for point in sorted(
            set(old_tokens.files) & set(new_tokens.files),
            key=lambda name: int(name.split("nfe")[-1]),
        ):
            if not point.startswith("argmax_"):
                continue
            old_array, new_array = old_tokens[point], new_tokens[point]
            changed = old_array != new_array
            drift_rows.append({
                "model": model,
                "point": point,
                "steps_per_block": int(point.split("nfe")[-1]),
                "n_sequences": old_array.shape[0],
                "sequence_length": old_array.shape[1],
                "token_agreement": float((~changed).mean()),
                "exact_sequence_fraction": float((~changed).all(axis=1).mean()),
                "mean_changed_tokens_per_sequence": float(changed.sum(axis=1).mean()),
                "old_tokens": str(old_path),
                "new_tokens": str(new_path),
            })
    drift_frame = pd.DataFrame(drift_rows)
    drift_path = args.output_dir / "m2_m3_token_drift.csv"
    drift_frame.to_csv(drift_path, index=False)

    _quality_plot(curves, "mauve", args.output_dir / "mauve_vs_optimized_latency.png")
    _quality_plot(curves, "gen_ppl", args.output_dir / "genppl_vs_optimized_latency.png")
    _speedup_plot(latency, args.output_dir / "speedup_vs_previous_fast.png")

    latency_table = latency[
        ((latency.model.isin(["M2", "M3"])) & (latency.discretize == "argmax"))
        | ((latency.model == "BD3-LM") & ~latency.sampler.str.contains("first_hitting"))
    ].pivot(index="steps_per_block", columns="model", values="sequence_latency_ms")
    latency_table = latency_table.reindex([1, 2, 4, 8, 16, 32])
    latency_table.columns.name = None
    latency_table.index.name = "steps/block"
    report = [
        "# Fused sampler H100 measurements",
        "",
        "Protocol: NVIDIA H100 NVL, FP32, batch 1, length 256, block size 16, "
        "5 warmups + 30 measured repeats; timing functions loaded from upstream commit "
        "`2e21c57704ab94e131e3350ffd8286632f61a7b8`.",
        "",
        "## Median sequence latency (ms)",
        "",
        _markdown_latency_table(latency_table),
        "",
        "BD3-LM first-hitting (256 total denoising NFEs): "
        f"{float(latency[latency.sampler.str.contains('first_hitting')].sequence_latency_ms.iloc[0]):.3f} ms.",
        "",
        "## Review notes",
        "",
        "- Fusing the clean-prefix commit with the next block's first denoising call removes all separate cache-maintenance forwards.",
        "- The optimized paths make exactly `16 × steps/block` backbone calls for ancestral/flow sampling; first-hitting makes 256 calls.",
        "- Literal in-place preallocated KV was not selected: PyTorch reported that mutated inputs disabled CUDA Graph capture; at M3 NFE/block=1 it took "
        f"{ablation['results']['preallocated']['sequence_latency_ms']:.3f} ms vs "
        f"{ablation['results']['dynamic']['sequence_latency_ms']:.3f} ms for dynamic fused KV "
        f"({ablation['preallocated_over_dynamic_slowdown']:.3f}× slower).",
        "- Therefore the fastest measured path still uses dynamic K/V concatenation with PyTorch SDPA. A true FlashAttention KV-cache kernel is not claimed by these measurements.",
        "- M2/M3 quality was rescored because compiled SDPA ordering introduces small FP differences. BD3-LM quality is reused only after exact canonical-vs-fast token parity checks.",
        f"- Across M2/M3 argmax dumps, token agreement with the previous fast path is {drift_frame.token_agreement.min():.3%}–{drift_frame.token_agreement.max():.3%}; the largest absolute MAUVE change is {deltas_frame.delta_mauve.abs().max():.4f}. These are real numerical-path differences, so the new latency is paired only with the rescored quality.",
        f"- BD3-LM parity is exact at all seven measured points (seed 0, batch 1), stored in `{parity_path}`.",
        "- MAUVE and genPPL use the existing 512-sample canonical protocol (except the separately identified 2048-sample BD3 headline, which is not used here).",
        "",
        "## Artifacts",
        "",
        f"- `{curves_path}`",
        f"- `{deltas_path}`",
        f"- `{drift_path}`",
        f"- `{ablation_path}`",
        f"- `{parity_path}`",
        f"- `{args.output_dir / 'mauve_vs_optimized_latency.png'}`",
        f"- `{args.output_dir / 'genppl_vs_optimized_latency.png'}`",
        f"- `{args.output_dir / 'speedup_vs_previous_fast.png'}`",
    ]
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(
        f"[done] rows={len(curves)} curves={curves_path} deltas={deltas_path} "
        f"report={report_path}", flush=True,
    )


if __name__ == "__main__":
    main()
