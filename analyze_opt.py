"""Final comparison, with every optimization enabled on both sides.

Optimizations are graded by what they cost in output, because "fastest" is only
meaningful next to "unchanged":

* **lossless** — provably the same computation. Verified token-for-token against
  each repo's own untouched sampler at the same seed:
  - CFM: the sparse-vocab sampler plus the clean-prefix KV cache (`cfm_fast.py`).
  - BD3-LM: `use_kv_cache=True`, which its `generate()` already implements,
    plus `torch.compile` on the backbone (also verified token-identical).
  `torch.compile` is not wired into CFM's cached path: the cache grows a block
  at a time, so the shapes are dynamic, and the path is already small enough
  that the win would come from CUDA graphs rather than fusion.
* **bf16** — reduced precision. Reported separately: the block-causal sampler is
  chaotic under perturbation (bf16 moves ~55% of CFM's tokens and ~38% of
  BD3-LM's), so it yields a *different* sample of the same quality, not the same
  sample faster.

    python "eval test/analyze_opt.py"
"""

from __future__ import annotations

import argparse
import json

from common import RESULTS


def rows(name: str, key: str) -> dict[tuple[int, int], dict]:
    try:
        data = json.loads((RESULTS / name).read_text())
    except FileNotFoundError:
        return {}
    return {(r["batch_size"], r[key]): r for r in data["rows"]
            if r.get("sampler") != "first_hitting"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="comparison_optimized.md")
    args = parser.parse_args()

    cfm = {
        "baseline": rows("cfm_latency.json", "nfe"),
        "lossless": rows("cfm_latency_lossless.json", "nfe"),
        "bf16": rows("cfm_latency_max.json", "nfe"),
    }
    bd3 = {
        "baseline": rows("bd3lm_latency_nocache.json", "steps_per_block"),
        "lossless": rows("bd3lm_latency_lossless.json", "steps_per_block"),
        "bf16": rows("bd3lm_latency_opt.json", "steps_per_block"),
    }
    # BD3-LM's cache-on/fp32 run predates the --compile flag; it is the fallback
    # for any point the compiled sweep did not reach.
    cache_only = rows("bd3lm_latency.json", "steps_per_block")
    for point, value in cache_only.items():
        bd3["lossless"].setdefault(point, value)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    meta = json.loads((RESULTS / "cfm_latency_lossless.json").read_text())["meta"]
    emit("# Every optimization enabled, both models")
    emit()
    emit(f"- GPU **{meta['gpu']}**, torch {meta['torch']}, L=256, block_size 16, "
         "256-token sequences.")
    emit("- **baseline** = each repo's untouched sampler, fp32, no caching.")
    emit("- **lossless** = verified token-for-token identical to that baseline: "
         "CFM sparse-vocab + clean-KV cache (fp32, eager); BD3-LM "
         "`use_kv_cache=True` + `torch.compile`.")
    emit("- **bf16** = the same, in bfloat16. A different sample, not a faster "
         "identical one.")
    emit()

    batches = sorted({b for b, _ in cfm["lossless"]} & {b for b, _ in bd3["lossless"]})
    steps = sorted({s for _, s in cfm["lossless"]})

    emit("## ms per sequence — lossless on both sides")
    emit()
    for batch in batches:
        emit(f"### batch = {batch}")
        emit()
        emit("| steps/block | CFM base | CFM lossless | gain | BD3-LM base "
             "| BD3-LM lossless | gain | **winner** |")
        emit("|---|---|---|---|---|---|---|---|")
        for s in steps:
            cb, cl = cfm["baseline"].get((batch, s)), cfm["lossless"].get((batch, s))
            bb, bl = bd3["baseline"].get((batch, s)), bd3["lossless"].get((batch, s))
            if not cl or not bl:
                continue
            cgain = (f"{cb['latency_s_per_batch'] / cl['latency_s_per_batch']:.1f}x"
                     if cb else "n/a (OOM)")
            bgain = (f"{bb['latency_s_per_batch'] / bl['latency_s_per_batch']:.1f}x"
                     if bb else "–")
            ratio = bl["latency_s_per_batch"] / cl["latency_s_per_batch"]
            winner = (f"**CFM {ratio:.1f}x**" if ratio > 1 else f"**BD3-LM {1/ratio:.1f}x**")
            cb_ms = f"{cb['latency_ms_per_sequence']:.1f}" if cb else "OOM"
            bb_ms = f"{bb['latency_ms_per_sequence']:.1f}" if bb else "–"
            emit(f"| {s} | {cb_ms} | **{cl['latency_ms_per_sequence']:.2f}** | {cgain} "
                 f"| {bb_ms} | **{bl['latency_ms_per_sequence']:.2f}** | {bgain} "
                 f"| {winner} |")
        emit()

    emit("## Does bf16 buy anything on top?")
    emit()
    emit("| batch | steps | CFM lossless | CFM bf16 | BD3-LM lossless | BD3-LM bf16 |")
    emit("|---|---|---|---|---|---|")
    for batch in batches:
        for s in steps:
            cl, cm = cfm["lossless"].get((batch, s)), cfm["bf16"].get((batch, s))
            bl, bm = bd3["lossless"].get((batch, s)), bd3["bf16"].get((batch, s))
            if not (cl and cm and bl and bm) or s not in (1, 4, 16):
                continue
            emit(f"| {batch} | {s} | {cl['latency_ms_per_sequence']:.2f} | "
                 f"{cm['latency_ms_per_sequence']:.2f} | "
                 f"{bl['latency_ms_per_sequence']:.2f} | "
                 f"{bm['latency_ms_per_sequence']:.2f} |")
    emit()

    emit("## Peak memory, GB (steps/block = 4)")
    emit()
    emit("| batch | CFM base | CFM lossless | BD3-LM base | BD3-LM lossless |")
    emit("|---|---|---|---|---|")
    for batch in batches:
        cb, cl = cfm["baseline"].get((batch, 4)), cfm["lossless"].get((batch, 4))
        bb, bl = bd3["baseline"].get((batch, 4)), bd3["lossless"].get((batch, 4))
        if not cl or not bl:
            continue
        emit(f"| {batch} | {cb['peak_mem_gb'] if cb else 'OOM'} | {cl['peak_mem_gb']} | "
             f"{bb['peak_mem_gb'] if bb else '–'} | {bl['peak_mem_gb']} |")
    emit()

    emit("## Throughput ceiling over the whole grid")
    emit()
    emit("| model | regime | best tok/s | at |")
    emit("|---|---|---|---|")
    for name, tables in [("CFM", cfm), ("BD3-LM", bd3)]:
        for regime in ("baseline", "lossless", "bf16"):
            table = tables[regime]
            if not table:
                continue
            best = max(table.values(), key=lambda r: r["tokens_per_sec"])
            emit(f"| {name} | {regime} | **{best['tokens_per_sec']:,.0f}** | batch "
                 f"{best['batch_size']}, {best.get('nfe', best.get('steps_per_block'))} steps |")
    emit()

    (RESULTS / args.out).write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {RESULTS / args.out}")


if __name__ == "__main__":
    main()
