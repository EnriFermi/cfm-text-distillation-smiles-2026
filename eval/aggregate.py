"""Turn versioned results/<exp>/metrics.json into the paper's tables and figures.

Run after a sweep::

    python -m eval.aggregate

Writes:
- ``results/summary.csv``            one row per run (seeds included)
- ``results/figures/*.png``          the headline figures (if matplotlib is present)

Figures kept: gen-PPL / NLL vs total network forwards, sequence latency, and
tokens/sec vs steps/block. A ``sampler=gold`` run adds the data-reference line
to quality plots. Numbers are averaged over seeds with std as error bars.
Kept dependency-light: CSV always, plots only if matplotlib is importable.
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
FIELDS = ["metric_schema", "model", "sampler", "block_size", "steps_per_block",
          "flow_nfe_total", "flow_nfe_per_token", "cache_encode_forwards",
          "nfe_total", "nfe_per_token", "context_token_cost_per_token", "cost_model",
          "seed", "prefix", "discretize", "schedule", "gen_ppl", "tokens_per_sec",
          "sampling_seconds", "sequence_latency_ms", "sequence_latency_p10_ms",
          "sequence_latency_p90_ms", "sequence_latency_repeats", "latency_device",
          "report_block_size", "n_samples", "commit"]

# Only these plots are generated; anything else under figures/ is removed on aggregate.
KEEP_FIGURES = {
    "nfe_total_vs_genppl.png",
    "nfe_total_vs_nll.png",
    "sequence_latency.png",
    "tokens_per_sec.png",
}


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("*/metrics.json")):
        d = json.loads(f.read_text())
        if d.get("metric_schema") != METRIC_SCHEMA:
            continue
        d["_exp"] = f.parent.name
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


def _agg(rows, key_fields, x_field, y_field="gen_ppl", where=None):
    """Group rows, averaging y over seeds. Returns sorted [(x, mean, std, group), ...]."""
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
    """Mean quality of ``sampler=gold`` runs (the data floor), if present."""
    vals = [r[y_field] for r in rows if r.get("sampler") == "gold" and r.get(y_field)]
    return st.mean(vals) if vals else None


def _with_gen_nll(rows: list[dict]) -> list[dict]:
    """Attach ``gen_nll = ln(gen_ppl)`` (nats); judge PPL is defined as ``exp(NLL)``."""
    out = []
    for r in rows:
        d = dict(r)
        if d.get("gen_ppl"):
            d["gen_nll"] = math.log(float(d["gen_ppl"]))
        out.append(d)
    return out


def _recipe_label(r: dict) -> str:
    return "M1" if r.get("model") == "M1" else f"M2 B{r['block_size']}"


def make_figures(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures (summary.csv still written)")
        return
    figdir = RESULTS / "figures"
    figdir.mkdir(exist_ok=True)
    rows_nll = _with_gen_nll(rows)
    model_rows = [r for r in rows if r.get("sampler") != "gold"]
    model_rows_nll = [r for r in rows_nll if r.get("sampler") != "gold"]
    ref_ppl = _data_reference(rows)
    ref_nll = _data_reference(rows_nll, y_field="gen_nll")

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

    # Quality vs total network forwards (PPL and NLL).
    _lineplot(_agg(model_rows, ["model", "block_size"], "nfe_total"),
              "network forwards / sequence", "nfe_total_vs_genppl.png",
              "Quality vs total network forwards", logx=True, grid=True,
              figsize=(7.5, 4.8), ref=ref_ppl)
    _lineplot(_agg(model_rows_nll, ["model", "block_size"], "nfe_total", y_field="gen_nll"),
              "network forwards / sequence", "nfe_total_vs_nll.png",
              "Quality vs total network forwards", logx=True, grid=True,
              figsize=(7.5, 4.8), ylabel="NLL (↓)", ref=ref_nll)

    # End-to-end batch-size-one latency. p10–p90 captures run-to-run jitter.
    # One point per steps_per_block (median over seeds if several measured).
    latency_rows = [
        r for r in model_rows
        if r.get("sequence_latency_ms") is not None
    ]
    if latency_rows:
        by_recipe: dict[str, list[dict]] = defaultdict(list)
        for r in latency_rows:
            by_recipe[_recipe_label(r)].append(r)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, rows_for_recipe in sorted(by_recipe.items()):
            # one point per steps_per_block: median over seeds if several measured
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
            ax.errorbar(
                xs, ys, yerr=[lower, upper], marker="o", capsize=3, label=label,
            )
        ax.set_xlabel("steps / block")
        ax.set_ylabel("latency (ms)")
        ax.set_title("End-to-end sequence latency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "sequence_latency.png", dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / 'sequence_latency.png'}")

    # Batched throughput vs steps/block (mean ± std over seeds).
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
        ax.set_title("Batched throughput vs steps/block")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figdir / "tokens_per_sec.png", dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / 'tokens_per_sec.png'}")

    for stale in figdir.glob("*.png"):
        if stale.name not in KEEP_FIGURES:
            stale.unlink()
            print(f"removed {stale}")


def main() -> None:
    rows = load_rows()
    if not rows:
        print(
            f"no {METRIC_SCHEMA} results found under {RESULTS}/*/metrics.json "
            "— run the current evaluation sweep first"
        )
        return
    write_csv(rows)
    make_figures(rows)


if __name__ == "__main__":
    main()
