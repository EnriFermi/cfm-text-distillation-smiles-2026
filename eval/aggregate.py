"""Turn versioned results/<exp>/metrics.json into tables and figures.

Run after a sweep::

    python -m eval.aggregate

Writes:
- ``results/summary.csv`` — all runs (Text8 + TinyStories)
- ``results/plot_points.csv`` (+ per-dataset CSV / ``plot_points.md``) — one row
  per figure point (model × B × steps/block) with gen-PPL, NLL, latency
- ``results/figures/text8/*.png`` — Text8-only plots
- ``results/figures/tinystories/*.png`` — TinyStories-only plots

Figures per dataset: gen-PPL / NLL vs total network forwards, sequence latency,
and tokens/sec vs steps/block. A ``sampler=gold`` run adds the data-reference
line to quality plots (Text8). Numbers are averaged over seeds with std bars.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
RESULTS = ROOT / "results"
METRIC_SCHEMA = "network_forwards_v2"
FIELDS = ["metric_schema", "dataset", "model", "sampler", "block_size", "steps_per_block",
          "flow_nfe_total", "flow_nfe_per_token", "cache_encode_forwards",
          "nfe_total", "nfe_per_token", "context_token_cost_per_token", "cost_model",
          "seed", "prefix", "discretize", "schedule", "gen_ppl", "tokens_per_sec",
          "sampling_seconds", "sequence_latency_ms", "sequence_latency_p10_ms",
          "sequence_latency_p90_ms", "sequence_latency_repeats", "latency_device",
          "report_block_size", "n_samples", "commit"]

PLOT_NAMES = (
    "nfe_total_vs_genppl.png",
    "nfe_total_vs_nll.png",
    "sequence_latency.png",
    "tokens_per_sec.png",
)

# Models that belong to TinyStories even if ``dataset`` was not written yet.
_TS_MODELS = {"M1-TS", "M2-TS", "AR-TS", "MDLM-TS", "BD3LM-TS"}


def _infer_dataset(d: dict, exp_name: str) -> str:
    if d.get("dataset") in ("text8", "tinystories"):
        return d["dataset"]
    if d.get("model") in _TS_MODELS or exp_name.startswith("ts_"):
        return "tinystories"
    return "text8"


def _display_model(model: str) -> str:
    """Drop ``-TS`` suffix on TinyStories plots so legends stay short."""
    return model[:-3] if isinstance(model, str) and model.endswith("-TS") else model


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("*/metrics.json")):
        d = json.loads(f.read_text())
        if d.get("metric_schema") != METRIC_SCHEMA:
            continue
        d["_exp"] = f.parent.name
        d["dataset"] = _infer_dataset(d, f.parent.name)
        rows.append(d)
    return rows


def write_csv(rows: list[dict]) -> None:
    out = RESULTS / "summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["_exp", *FIELDS], extrasaction="ignore", lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt_num(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "—"
    if nd == 0:
        return f"{x:.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def _fmt_pm(mean: float | None, std: float | None, nd: int = 2) -> str:
    if mean is None:
        return "—"
    if std is None or std == 0:
        return _fmt_num(mean, nd)
    return f"{_fmt_num(mean, nd)} ± {_fmt_num(std, nd)}"


def write_plot_points(rows: list[dict]) -> None:
    """One row per figure point: model × block_size × steps_per_block.

    Matches ``results/figures/*/`` aggregation: gen-PPL / NLL mean±std over seeds,
    latency median (batch=1). Always includes ``steps_per_block``.
    """
    model_rows = [r for r in rows if r.get("sampler") != "gold"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in model_rows:
        model = _display_model(r.get("model"))
        key = (
            r["dataset"],
            model,
            int(r["block_size"]),
            int(r["steps_per_block"]),
        )
        d = dict(r)
        d["model"] = model
        if d.get("gen_ppl") is not None and d.get("gen_nll") is None:
            d["gen_nll"] = math.log(float(d["gen_ppl"]))
        groups[key].append(d)

    point_fields = [
        "dataset", "model", "recipe", "block_size", "steps_per_block", "nfe_total",
        "n_seeds_ppl", "gen_ppl_mean", "gen_ppl_std", "gen_nll_mean", "gen_nll_std",
        "n_seeds_latency", "sequence_latency_ms_median",
        "sequence_latency_p10_ms_median", "sequence_latency_p90_ms_median",
    ]
    out_rows: list[dict] = []
    for (dataset, model, block_size, steps), pts in sorted(
        groups.items(), key=lambda t: (t[0][0], t[0][1], t[0][2], t[0][3])
    ):
        ppls = [float(p["gen_ppl"]) for p in pts if p.get("gen_ppl") is not None]
        nlls = [
            float(p["gen_nll"]) if p.get("gen_nll") is not None
            else math.log(float(p["gen_ppl"]))
            for p in pts if p.get("gen_ppl") is not None
        ]
        nfes = [float(p["nfe_total"]) for p in pts if p.get("nfe_total") is not None]
        lats = [
            float(p["sequence_latency_ms"])
            for p in pts if p.get("sequence_latency_ms") is not None
        ]
        lat_p10 = [
            float(p["sequence_latency_p10_ms"])
            for p in pts if p.get("sequence_latency_p10_ms") is not None
        ]
        lat_p90 = [
            float(p["sequence_latency_p90_ms"])
            for p in pts if p.get("sequence_latency_p90_ms") is not None
        ]
        ppl_m, ppl_s = _mean_std(ppls) if ppls else (None, None)
        nll_m, nll_s = _mean_std(nlls) if nlls else (None, None)
        recipe = _recipe_label({"model": model, "block_size": block_size})
        out_rows.append({
            "dataset": dataset,
            "model": model,
            "recipe": recipe,
            "block_size": block_size,
            "steps_per_block": steps,
            "nfe_total": st.mean(nfes) if nfes else None,
            "n_seeds_ppl": len(ppls),
            "gen_ppl_mean": ppl_m,
            "gen_ppl_std": ppl_s,
            "gen_nll_mean": nll_m,
            "gen_nll_std": nll_s,
            "n_seeds_latency": len(lats),
            "sequence_latency_ms_median": st.median(lats) if lats else None,
            "sequence_latency_p10_ms_median": st.median(lat_p10) if lat_p10 else None,
            "sequence_latency_p90_ms_median": st.median(lat_p90) if lat_p90 else None,
        })

    combined = RESULTS / "plot_points.csv"
    with combined.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=point_fields, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {combined} ({len(out_rows)} points)")

    for dataset in ("text8", "tinystories"):
        subset = [r for r in out_rows if r["dataset"] == dataset]
        path = RESULTS / f"plot_points_{dataset}.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=point_fields, lineterminator="\n")
            w.writeheader()
            w.writerows(subset)
        print(f"wrote {path} ({len(subset)} points)")

    md_lines = [
        "# Plot points (aggregated over seeds)",
        "",
        "Same grouping as `results/figures/{text8,tinystories}/`.",
        "`gen_ppl` / `gen_nll`: mean ± std over seeds; `latency`: median ms (batch=1, L=256).",
        "`gen_nll = log(gen_ppl)`. Regenerated by `python -m eval.aggregate`.",
        "",
    ]
    for dataset, title in (("text8", "Text8"), ("tinystories", "TinyStories")):
        subset = [r for r in out_rows if r["dataset"] == dataset]
        if not subset:
            continue
        md_lines.extend([
            f"## {title}",
            "",
            "| model | B | steps/block | NFE/seq | gen-PPL ↓ | NLL ↓ | latency ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for r in subset:
            md_lines.append(
                f"| {r['recipe']} | {r['block_size']} | {r['steps_per_block']} | "
                f"{_fmt_num(r['nfe_total'], 0)} | "
                f"{_fmt_pm(r['gen_ppl_mean'], r['gen_ppl_std'], 1)} | "
                f"{_fmt_pm(r['gen_nll_mean'], r['gen_nll_std'], 3)} | "
                f"{_fmt_num(r['sequence_latency_ms_median'], 1)} |"
            )
        md_lines.append("")
    md_path = RESULTS / "plot_points.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {md_path}")

    (RESULTS / "plot_points.json").write_text(json.dumps(out_rows, indent=2) + "\n")
    print(f"wrote {RESULTS / 'plot_points.json'}")


def _agg(rows, key_fields, x_field, y_field="gen_ppl", where=None):
    groups = defaultdict(list)
    for r in rows:
        if r.get(y_field) is None or r.get(x_field) is None:
            continue
        if where and not where(r):
            continue
        gkey = tuple(r.get(k) for k in key_fields)
        groups[(gkey, r.get(x_field))].append(r[y_field])
    out = []
    for (gkey, x), ys in groups.items():
        m, s = _mean_std(ys)
        out.append((x, m, s, gkey))
    return sorted(out, key=lambda t: (str(t[3]), t[0]))


def _data_reference(rows, *, y_field: str = "gen_ppl") -> float | None:
    vals = [r[y_field] for r in rows if r.get("sampler") == "gold" and r.get(y_field)]
    return st.mean(vals) if vals else None


def _with_gen_nll(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        if d.get("gen_ppl"):
            d["gen_nll"] = math.log(float(d["gen_ppl"]))
        out.append(d)
    return out


def _recipe_label(r: dict) -> str:
    model = _display_model(r.get("model"))
    if model == "M1":
        return "M1"
    if model == "M2":
        return f"M2 B{r['block_size']}"
    if model == "M3":
        return f"M3 B{r['block_size']}"
    if model in ("AR", "MDLM", "BD3LM"):
        b = r.get("block_size")
        return f"{model}" + (f" B{b}" if model == "BD3LM" and b else "")
    return f"{model} B{r.get('block_size')}"


def make_figures_for_dataset(rows: list[dict], dataset: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures")
        return

    figdir = RESULTS / "figures" / dataset
    figdir.mkdir(parents=True, exist_ok=True)

    # Normalize model names for grouping/legends within this dataset.
    normed = []
    for r in rows:
        d = dict(r)
        d["model"] = _display_model(d.get("model"))
        normed.append(d)

    rows_nll = _with_gen_nll(normed)
    model_rows = [r for r in normed if r.get("sampler") != "gold"]
    model_rows_nll = [r for r in rows_nll if r.get("sampler") != "gold"]
    ref_ppl = _data_reference(normed)
    ref_nll = _data_reference(rows_nll, y_field="gen_nll")
    title_ds = "Text8" if dataset == "text8" else "TinyStories"

    def _lineplot(agg, xlabel, fname, title, logx=False, *, grid=False,
                  figsize=(6.5, 4.5), ylabel="gen-PPL (↓)", ref=None):
        by_group = defaultdict(list)
        for x, m, s, g in agg:
            by_group[g].append((x, m, s))
        if not by_group:
            return
        fig, ax = plt.subplots(figsize=figsize)
        for g, pts in by_group.items():
            pts.sort()
            xs, ms, ss = zip(*pts)
            label = "-".join(str(x) for x in g) if isinstance(g, tuple) else str(g)
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, label=label)
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1, color="gray", label="data")
        if logx:
            ax.set_xscale("log")
        if grid:
            ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / fname}")

    _lineplot(_agg(model_rows, ["model", "block_size"], "nfe_total"),
              "network forwards / sequence", "nfe_total_vs_genppl.png",
              f"{title_ds}: quality vs network forwards", logx=True, grid=True,
              figsize=(7.5, 4.8), ref=ref_ppl)
    _lineplot(_agg(model_rows_nll, ["model", "block_size"], "nfe_total", y_field="gen_nll"),
              "network forwards / sequence", "nfe_total_vs_nll.png",
              f"{title_ds}: quality vs network forwards", logx=True, grid=True,
              figsize=(7.5, 4.8), ylabel="NLL (↓)", ref=ref_nll)

    latency_rows = [r for r in model_rows if r.get("sequence_latency_ms") is not None]
    if latency_rows:
        by_recipe: dict[str, list[dict]] = defaultdict(list)
        for r in latency_rows:
            by_recipe[_recipe_label(r)].append(r)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, rows_for_recipe in sorted(by_recipe.items()):
            by_steps: dict[int, list[dict]] = defaultdict(list)
            for r in rows_for_recipe:
                by_steps[int(r["steps_per_block"])].append(r)
            xs, ys, lower, upper = [], [], [], []
            for steps in sorted(by_steps):
                pts = by_steps[steps]
                y = st.median([p["sequence_latency_ms"] for p in pts])
                p10 = st.median([p.get("sequence_latency_p10_ms", y) for p in pts])
                p90 = st.median([p.get("sequence_latency_p90_ms", y) for p in pts])
                xs.append(steps)
                ys.append(y)
                lower.append(y - p10)
                upper.append(p90 - y)
            ax.errorbar(xs, ys, yerr=[lower, upper], marker="o", capsize=3, label=label)
        ax.set_xlabel("steps / block")
        ax.set_ylabel("latency (ms)")
        ax.set_yscale("log")
        ax.set_title(f"{title_ds}: end-to-end sequence latency")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figdir / "sequence_latency.png", dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / 'sequence_latency.png'}")

    tps_rows = [r for r in model_rows if r.get("tokens_per_sec") is not None]
    if tps_rows:
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        for r in tps_rows:
            buckets[(_recipe_label(r), int(r["steps_per_block"]))].append(
                float(r["tokens_per_sec"])
            )
        by_recipe_tps: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for (label, steps), vals in buckets.items():
            m, s = _mean_std(vals)
            by_recipe_tps[label].append((steps, m, s))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, pts in sorted(by_recipe_tps.items()):
            pts.sort()
            xs, ms, ss = zip(*pts)
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, label=label)
        ax.set_xlabel("steps / block")
        ax.set_ylabel("tokens / sec")
        ax.set_title(f"{title_ds}: batched throughput vs steps/block")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figdir / "tokens_per_sec.png", dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / 'tokens_per_sec.png'}")

    keep = set(PLOT_NAMES)
    for stale in figdir.glob("*.png"):
        if stale.name not in keep:
            stale.unlink()
            print(f"removed {stale}")


def make_figures(rows: list[dict]) -> None:
    fig_root = RESULTS / "figures"
    fig_root.mkdir(exist_ok=True)
    # Drop legacy mixed plots sitting directly under figures/
    for stale in fig_root.glob("*.png"):
        stale.unlink()
        print(f"removed legacy mixed plot {stale}")

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for dataset in ("text8", "tinystories"):
        subset = by_ds.get(dataset, [])
        if not subset:
            print(f"no rows for dataset={dataset}; skipping figures")
            continue
        print(f"=== figures for {dataset} ({len(subset)} runs) ===")
        make_figures_for_dataset(subset, dataset)


def main() -> None:
    rows = load_rows()
    if not rows:
        print(
            f"no {METRIC_SCHEMA} results found under {RESULTS}/*/metrics.json "
            "— run the current evaluation sweep first"
        )
        return
    write_csv(rows)
    write_plot_points(rows)
    make_figures(rows)


if __name__ == "__main__":
    main()
