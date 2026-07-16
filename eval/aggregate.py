"""Turn results/<exp>/metrics.json into the paper's tables and figures.

Run after a sweep::

    python -m eval.aggregate

Writes:
- ``results/summary.csv``            one row per run (seeds included)
- ``results/figures/*.png``          the headline figures (if matplotlib is present)

Figures map to docs/experiment_plan.md: Fig.1 gen-PPL vs NFE/token (E1), Fig.1b vs
FLOP cost, Fig.2 vs block size (E3), Fig.3 vs steps/block (E4), Fig.4 entropy per
block index (E5), Fig.5 vs wall-clock tokens/sec (E13). A ``sampler=gold`` run adds
the data-reference line to quality plots. Numbers are averaged over seeds with std as
error bars. Kept dependency-light: CSV always, plots only if matplotlib is importable.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
RESULTS = ROOT / "results"
FIELDS = ["model", "sampler", "block_size", "steps_per_block", "nfe_per_token",
          "flop_cost_per_token", "cost_model", "seed", "prefix",
          "discretize", "schedule", "gen_ppl", "tokens_per_sec", "sampling_seconds",
          "report_block_size", "n_samples", "commit"]


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("*/metrics.json")):
        d = json.loads(f.read_text())
        d["_exp"] = f.parent.name
        rows.append(d)
    return rows


def write_csv(rows: list[dict]) -> None:
    out = RESULTS / "summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["_exp", *FIELDS], extrasaction="ignore")
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


def _data_reference(rows) -> float | None:
    """Mean gen-PPL of ``sampler=gold`` runs (the data floor), if present."""
    vals = [r["gen_ppl"] for r in rows if r.get("sampler") == "gold" and r.get("gen_ppl")]
    return st.mean(vals) if vals else None


def _is_legacy_pin_run(row: dict) -> bool:
    """Exclude deprecated pin-prefix M2 runs from headline figures."""
    model = str(row.get("model", ""))
    if model.endswith("-pin"):
        return True
    return row.get("use_mask") is False  # old metrics.json field


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
    model_rows = [r for r in rows if r.get("sampler") != "gold" and not _is_legacy_pin_run(r)]
    ref_ppl = _data_reference(rows)

    def _lineplot(agg, xlabel, fname, title, logx=False):
        by_group = defaultdict(list)
        for x, m, s, g in agg:
            by_group[g].append((x, m, s))
        if not by_group:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        for g, pts in by_group.items():
            pts.sort()
            xs, ms, ss = zip(*pts)
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, label=str(g))
        if ref_ppl is not None:
            ax.axhline(ref_ppl, ls="--", lw=1, color="gray", label="data")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("gen-PPL (↓)"); ax.set_title(title)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(figdir / fname, dpi=150); plt.close(fig)
        print(f"wrote {figdir / fname}")

    # Fig.1 — gen-PPL vs NFE/token, one series per model (E1, H2)
    _lineplot(_agg(model_rows, ["model"], "nfe_per_token"),
              "NFE / token", "fig1_nfe_vs_genppl.png", "Quality vs NFE", logx=True)
    # Fig.1b — same on the honest FLOP axis (full-recompute cost; see block/nfe.py)
    _lineplot(_agg(model_rows, ["model"], "flop_cost_per_token"),
              "token-passes / token", "fig1b_flops_vs_genppl.png", "Quality vs FLOP cost", logx=True)
    # Fig.2 — gen-PPL vs block size at steps/block=1 (E3, sweet spot)
    _lineplot(_agg(model_rows, ["model"], "block_size", where=lambda r: r.get("steps_per_block") == 1),
              "block size B", "fig2_blocksize.png", "Quality vs block size (1 step/block)", logx=True)
    # Fig.3 — gen-PPL vs steps/block at B=16 (E4)
    _lineplot(_agg(model_rows, ["model"], "steps_per_block", where=lambda r: r.get("block_size") == 16),
              "steps / block", "fig3_steps.png", "Quality vs steps/block (B=16)")
    # Fig.5 — gen-PPL vs wall-clock throughput (E13)
    _lineplot(_agg(model_rows, ["model"], "tokens_per_sec"),
              "tokens / sec", "fig5_speed_vs_genppl.png", "Quality vs throughput", logx=True)

    # Fig.4 — per-sample entropy per block index (E5, collapse), one line per run
    curves = [r for r in model_rows if len(r.get("entropy_per_block_ps", [])) > 1]
    if curves:
        fig, ax = plt.subplots(figsize=(5, 4))
        for r in curves:
            ax.plot(range(len(r["entropy_per_block_ps"])), r["entropy_per_block_ps"],
                    marker=".", label=f"{r.get('model')} B{r.get('report_block_size')}")
        ax.set_xlabel("block index"); ax.set_ylabel("token entropy / sample (nats)")
        ax.set_title("Entropy collapse across blocks"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / "fig4_entropy_per_block.png", dpi=150)
        plt.close(fig); print(f"wrote {figdir / 'fig4_entropy_per_block.png'}")


def main() -> None:
    rows = load_rows()
    if not rows:
        print(f"no results found under {RESULTS}/*/metrics.json — run eval.run_eval first")
        return
    write_csv(rows)
    make_figures(rows)


if __name__ == "__main__":
    main()
