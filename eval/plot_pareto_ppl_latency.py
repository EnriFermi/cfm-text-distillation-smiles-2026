"""Pareto-style scatter: gen-PPL vs sequence latency (batch=1, L=256).

One point per (model, block_size, steps). PPL = mean over seeds; latency from
the seed-0 latency benchmark. Points are not connected; each has a boxed
mini-label (model, bs, nfe).
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "pareto_ppl_vs_latency.png"
OUT_ALL = RESULTS / "figures" / "pareto_ppl_vs_latency_all.png"


def _label(model: str, block_size, steps) -> str:
    if model == "AR":
        return "AR"
    if model == "MDLM":
        return f"MDLM, nfe={steps}"
    if model == "M1":
        return f"M1, nfe={steps}"
    return f"{model}, bs={block_size}, nfe={steps}"


def _collect() -> list[dict]:
    ppls: dict[tuple, list[float]] = defaultdict(list)
    lats: dict[tuple, float] = {}
    for path in RESULTS.glob("*/metrics.json"):
        m = json.loads(path.read_text())
        if m.get("metric_schema") != "network_forwards_v2":
            continue
        if m.get("sampler") == "gold":
            continue
        if m.get("gen_ppl") is None:
            continue
        key = (m["model"], m.get("block_size"), m.get("steps_per_block"))
        ppls[key].append(float(m["gen_ppl"]))
        if m.get("sequence_latency_ms") is not None and int(m.get("seed", 0)) == 0:
            lats[key] = float(m["sequence_latency_ms"])

    rows = []
    for key, vals in ppls.items():
        if key not in lats:
            continue
        model, B, steps = key
        rows.append({
            "model": model,
            "block_size": B,
            "steps": steps,
            "ppl": st.mean(vals),
            "latency_ms": lats[key],
            "label": _label(model, B, steps),
        })
    return rows


def _main_comparison(rows: list[dict]) -> list[dict]:
    """Readable subset: AR, M1, M2/M3/BD3 at B=16, MDLM."""
    out = []
    for r in rows:
        m, B = r["model"], r["block_size"]
        if m == "AR":
            out.append(r)
        elif m in ("M1", "MDLM"):
            out.append(r)
        elif m in ("M2", "M3", "BD3LM") and int(B) == 16:
            out.append(r)
    return out


def _pareto_mask(rows: list[dict]) -> list[bool]:
    pts = [(r["latency_ms"], r["ppl"]) for r in rows]
    out = []
    for i, (x, y) in enumerate(pts):
        dominated = any(
            (ox <= x and oy <= y) and (ox < x or oy < y)
            for j, (ox, oy) in enumerate(pts) if j != i
        )
        out.append(not dominated)
    return out


def _plot(rows: list[dict], out_path: Path, *, annotate_all: bool, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "AR": "#2ca02c",
        "MDLM": "#9467bd",
        "BD3LM": "#8c564b",
        "M1": "#1f77b4",
        "M2": "#ff7f0e",
        "M3": "#d62728",
    }
    markers = {
        "AR": "D",
        "MDLM": "s",
        "BD3LM": "^",
        "M1": "o",
        "M2": "o",
        "M3": "o",
    }

    on_front = _pareto_mask(rows)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    seen: set[str] = set()
    for r, front in zip(rows, on_front):
        model = r["model"]
        kw = dict(
            color=colors.get(model, "gray"),
            marker=markers.get(model, "o"),
            s=85 if front else 55,
            edgecolors="black",
            linewidths=1.0 if front else 0.55,
            zorder=3 if front else 2,
            alpha=0.95,
        )
        if model not in seen:
            ax.scatter(r["latency_ms"], r["ppl"], label=model, **kw)
            seen.add(model)
        else:
            ax.scatter(r["latency_ms"], r["ppl"], **kw)

    offsets = [
        (7, 9), (7, -13), (-8, 9), (-8, -15), (12, 2), (-16, 2),
        (9, 16), (9, -20), (-12, 14), (-12, -18), (14, -8), (-18, 10),
    ]
    for i, (r, front) in enumerate(zip(rows, on_front)):
        if not annotate_all and not front:
            continue
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            r["label"],
            xy=(r["latency_ms"], r["ppl"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.35",
                linewidth=0.6,
                alpha=0.92,
            ),
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.45),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("sequence latency (ms, batch=1, L=256)")
    ax.set_ylabel("gen-PPL (↓)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(title="model", fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path} ({len(rows)} points, {sum(on_front)} on front)")


def main() -> None:
    rows = _collect()
    if not rows:
        raise SystemExit("no points with both gen_ppl and sequence_latency_ms")
    main_rows = _main_comparison(rows)
    _plot(
        main_rows, OUT, annotate_all=True,
        title="Pareto view: gen-PPL vs latency (B=16 block methods + AR/M1/MDLM)",
    )
    _plot(
        rows, OUT_ALL, annotate_all=False,
        title="Pareto view: gen-PPL vs latency (all B; labels = front only)",
    )
    front = _pareto_mask(main_rows)
    print("main-comparison front:")
    for r, f in sorted(zip(main_rows, front), key=lambda t: t[0]["latency_ms"]):
        if f:
            print(f"  * {r['label']:28}  lat={r['latency_ms']:8.1f}  ppl={r['ppl']:7.1f}")


if __name__ == "__main__":
    main()
