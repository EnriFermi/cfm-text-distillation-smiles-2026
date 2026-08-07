"""Head-to-head at equal footing: neither model uses any inference caching.

CFM has no KV cache in any mode (``block_causal_sample`` rebuilds the whole
``[clean; noisy]`` 2L forward per jump), so the matching BD3-LM setting is
``use_kv_cache=False``. With the cache off BD3-LM also drops the per-block
cache-append forward, which makes its cost exactly proportional to
``steps_per_block`` — so the iso-latency fit here goes through the origin
instead of being affine.

Later ``--bd3lm`` files override earlier ones for the same (batch, S) point,
which is how the re-measured batch=4 column replaces its noisy first pass.

    python "eval test/analyze_fair.py"
"""

from __future__ import annotations

import argparse
import json

from common import RESULTS


def load_rows(names: list[str]) -> dict[tuple[int, int], dict]:
    merged: dict[tuple[int, int], dict] = {}
    for name in names:
        data = json.loads((RESULTS / name).read_text())
        for r in data["rows"]:
            if r["sampler"] != "ancestral":
                continue
            merged[(r["batch_size"], r["steps_per_block"])] = r
    return merged


def fit_through_origin(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope with the intercept pinned at 0: T(S) = alpha * S."""
    return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm", default="cfm_latency.json")
    parser.add_argument("--bd3lm", nargs="+",
                        default=["bd3lm_latency_nocache.json",
                                 "bd3lm_nocache_b4_rerun.json"])
    parser.add_argument("--bd3lm-cached", default="bd3lm_latency.json")
    parser.add_argument("--out", default="comparison_fair.md")
    args = parser.parse_args()

    cfm_data = json.loads((RESULTS / args.cfm).read_text())
    cfm = {(r["batch_size"], r["nfe"]): r for r in cfm_data["rows"]}
    bd3 = load_rows(args.bd3lm)
    cached = load_rows([args.bd3lm_cached])

    assert all(not r["kv_cache"] for r in bd3.values()), "expected cache-free rows"

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Equal-footing comparison: no inference caching on either side")
    emit()
    emit(f"- GPU: **{cfm_data['meta']['gpu']}**, torch {cfm_data['meta']['torch']}, fp32, L=256, block_size 16")
    emit("- CFM: `block_causal_sample` — a full `[clean; noisy]` 512-token forward per jump.")
    emit("- BD3-LM: `use_kv_cache=False`, ancestral sampler — the prefix is recomputed every step.")
    emit("- Both then run exactly `16 blocks x steps` network forwards per sequence.")
    emit()

    batches = sorted({b for b, _ in cfm} & {b for b, _ in bd3})
    steps = sorted({s for _, s in cfm.values() for s in [s]} if False else
                   {r["nfe"] for r in cfm_data["rows"]})

    emit("## ms per 256-token sequence, at equal steps-per-block")
    emit()
    header = "| steps/block |" + "".join(f" CFM b{b} | BD3-LM b{b} | ratio |" for b in batches)
    emit(header)
    emit("|---" * (1 + 3 * len(batches)) + "|")
    for s in steps:
        cells = ""
        for b in batches:
            c, d = cfm.get((b, s)), bd3.get((b, s))
            if not c or not d:
                cells += " – | – | – |"
                continue
            ratio = c["latency_s_per_batch"] / d["latency_s_per_batch"]
            cells += (f" {c['latency_ms_per_sequence']:.1f} | "
                      f"{d['latency_ms_per_sequence']:.1f} | **{ratio:.2f}x** |")
        emit(f"| {s} |{cells}")
    emit()
    emit("`ratio` > 1 means BD3-LM is that much faster at the same number of steps.")
    emit()

    emit("## Iso-latency, cache-free")
    emit()
    emit("| batch | BD3-LM cost `T(S)` | S matching CFM NFE=1 | =2 | =4 | =8 | =16 |")
    emit("|---|---|---|---|---|---|---|")
    for b in batches:
        pts = [(s, r) for (bb, s), r in bd3.items() if bb == b]
        alpha = fit_through_origin([s for s, _ in pts],
                                   [r["latency_s_per_batch"] for _, r in pts])
        cols = ""
        for nfe in steps:
            c = cfm.get((b, nfe))
            cols += f" {c['latency_s_per_batch'] / alpha:.1f} |" if c else " – |"
        emit(f"| {b} | {alpha:.4f}*S s/batch |{cols}")
    emit()

    emit("## What BD3-LM's KV cache is worth (same weights, same S)")
    emit()
    emit("| batch | steps/block | cache off ms/seq | cache on ms/seq | speedup |")
    emit("|---|---|---|---|---|")
    for b in batches:
        for s in steps:
            off, on = bd3.get((b, s)), cached.get((b, s))
            if not off or not on:
                continue
            emit(f"| {b} | {s} | {off['latency_ms_per_sequence']:.1f} | "
                 f"{on['latency_ms_per_sequence']:.1f} | "
                 f"**{off['latency_s_per_batch'] / on['latency_s_per_batch']:.2f}x** |")
    emit()

    (RESULTS / args.out).write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {RESULTS / args.out}")


if __name__ == "__main__":
    main()
