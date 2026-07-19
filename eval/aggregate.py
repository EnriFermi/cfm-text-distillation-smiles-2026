"""Turn versioned results/<exp>/metrics.json into the paper's tables and figures.

Run after a sweep::

    python -m eval.aggregate

Writes:
- ``results/summary.csv``            one row per run (seeds included)
- ``results/figures/*.png``          the headline figures (if matplotlib is present)

Figures map to docs/experiment_plan.md: Fig.1 gen-PPL vs NFE/token (E1), Fig.1b vs
FLOP cost, Fig.2 vs block size (E3), Fig.3 vs steps/block (E4), Fig.4 entropy per
block index (E5), Fig.5 vs wall-clock tokens/sec (E13), Fig.5a sequence latency,
Fig.5b tokens/sec vs steps/block. A ``sampler=gold`` run adds the data-reference
line to quality plots. Numbers are averaged over seeds with std as error bars.
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

    def _lineplot(agg, xlabel, fname, title, logx=False, *, grid=False, label_xticks=False,
                  figsize=(6.5, 4.5), ylabel="gen-PPL (↓)", ref=None):
        by_group = defaultdict(list)
        for x, m, s, g in agg:
            by_group[g].append((x, m, s))
        if not by_group:
            return
        fig, ax = plt.subplots(figsize=figsize)
        all_xs: list[float] = []
        for g, pts in by_group.items():
            pts.sort()
            xs, ms, ss = zip(*pts)
            all_xs.extend(xs)
            label = "-".join(str(x) for x in g) if isinstance(g, tuple) else str(g)
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, label=label)
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1, color="gray", label="data")
        if logx:
            ax.set_xscale("log")
        if label_xticks and all_xs:
            ticks = sorted({float(x) for x in all_xs})
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks], rotation=45, ha="right", fontsize=8)
        if grid:
            ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(figdir / fname, dpi=150); plt.close(fig)
        print(f"wrote {figdir / fname}")

    # Fig.1 — all network forwards per generated token, one series per recipe.
    _lineplot(_agg(model_rows, ["model", "block_size"], "nfe_per_token"),
              "network forwards / token", "fig1_nfe_vs_genppl.png",
              "Quality vs network forwards / token", logx=True, ref=ref_ppl)
    # Fig.1a — all network forwards per sequence; cache construction is included.
    _lineplot(_agg(model_rows, ["model", "block_size"], "nfe_total"),
              "network forwards / sequence", "fig1a_nfe_total_vs_genppl.png",
              "Quality vs total network forwards", logx=True, grid=True,
              figsize=(7.5, 4.8), ref=ref_ppl)
    # Fig.1a (NLL) — same axes, y = ln(gen_ppl) in nats.
    _lineplot(_agg(model_rows_nll, ["model", "block_size"], "nfe_total", y_field="gen_nll"),
              "network forwards / sequence", "fig1a_nfe_total_vs_nll.png",
              "Quality vs total network forwards", logx=True, grid=True,
              figsize=(7.5, 4.8), ylabel="NLL (↓)", ref=ref_nll)
    # Fig.1b — context-token proxy including clean-prefix cache construction.
    _lineplot(_agg(model_rows, ["model", "block_size"], "context_token_cost_per_token"),
              "context token-passes / output token (proxy)", "fig1b_flops_vs_genppl.png",
              "Quality vs context-token compute proxy", logx=True, ref=ref_ppl)
    # Fig.2 — gen-PPL vs block size at steps/block=1 (E3, sweet spot)
    _lineplot(_agg(model_rows, ["model"], "block_size", where=lambda r: r.get("steps_per_block") == 1),
              "block size B", "fig2_blocksize.png", "Quality vs block size (1 step/block)",
              logx=True, ref=ref_ppl)
    # Fig.3 — gen-PPL vs steps/block at B=16 (E4)
    _lineplot(_agg(model_rows, ["model"], "steps_per_block", where=lambda r: r.get("block_size") == 16),
              "steps / block", "fig3_steps.png", "Quality vs steps/block (B=16)", ref=ref_ppl)
    # Fig.5 — gen-PPL vs wall-clock throughput (E13)
    _lineplot(_agg(model_rows, ["model", "block_size"], "tokens_per_sec"),
              "tokens / sec", "fig5_speed_vs_genppl.png", "Quality vs throughput",
              logx=True, ref=ref_ppl)

    def _recipe_label(r: dict) -> str:
        return "M1" if r.get("model") == "M1" else f"M2 B{r['block_size']}"

    # Fig.5a — end-to-end batch-size-one latency. p10–p90 captures run-to-run jitter.
    latency_rows = [
        r for r in model_rows
        if r.get("sequence_latency_ms") is not None
    ]
    if latency_rows:
        by_recipe = defaultdict(list)
        for r in latency_rows:
            by_recipe[_recipe_label(r)].append(r)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, rows_for_recipe in sorted(by_recipe.items()):
            rows_for_recipe.sort(key=lambda r: r["steps_per_block"])
            xs = [r["steps_per_block"] for r in rows_for_recipe]
            ys = [r["sequence_latency_ms"] for r in rows_for_recipe]
            lower = [
                y - r.get("sequence_latency_p10_ms", y)
                for y, r in zip(ys, rows_for_recipe)
            ]
            upper = [
                r.get("sequence_latency_p90_ms", y) - y
                for y, r in zip(ys, rows_for_recipe)
            ]
            ax.errorbar(
                xs, ys, yerr=[lower, upper], marker="o", capsize=3,
                label=label,
            )
        ax.set_xlabel("steps / block")
        ax.set_ylabel("latency (ms)")
        ax.set_title("End-to-end sequence latency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "fig5a_sequence_latency.png", dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / 'fig5a_sequence_latency.png'}")

    # Fig.5b — batched throughput vs steps/block (same recipes as fig5a).
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
        fig.savefig(figdir / "fig5b_tokens_per_sec.png", dpi=150)
        plt.close(fig)
        print(f"wrote {figdir / 'fig5b_tokens_per_sec.png'}")

    # Fig.4 — per-sample entropy per block index (E5, collapse), one line per run
    curves = [r for r in model_rows if len(r.get("entropy_per_block_ps", [])) > 1]
    if curves:
        fig, ax = plt.subplots(figsize=(5, 4))
        for r in curves:
            ax.plot(range(len(r["entropy_per_block_ps"])), r["entropy_per_block_ps"],
                    marker=".", label=f"{r.get('model')} B{r.get('block_size')}")
        ax.set_xlabel("block index"); ax.set_ylabel("token entropy / sample (nats)")
        ax.set_title("Entropy collapse across blocks"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / "fig4_entropy_per_block.png", dpi=150)
        plt.close(fig); print(f"wrote {figdir / 'fig4_entropy_per_block.png'}")


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
