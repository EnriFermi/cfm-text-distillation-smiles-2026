"""Compare the two latency sweeps and solve for the iso-latency BD3-LM setting.

BD3-LM cost per block is affine in ``steps_per_block``: S denoiser forwards over
the active block plus one KV-cache append forward. So a least-squares fit of
``T(S) = alpha * S + beta`` on the measured points is the right interpolant, and
inverting it answers "which S makes BD3-LM as slow as CFM at this NFE".

    python "eval test/analyze.py"
"""

from __future__ import annotations

import argparse
import json

from common import RESULTS


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares slope/intercept — no numpy needed for two columns."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm", default="cfm_latency.json")
    parser.add_argument("--bd3lm", default="bd3lm_latency.json")
    parser.add_argument("--out", default="comparison.md")
    args = parser.parse_args()

    cfm = load(args.cfm)
    bd3 = load(args.bd3lm)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# CFM vs BD3-LM generation latency (TinyStories, L=256)")
    emit()
    emit(f"- GPU: **{cfm['meta']['gpu']}**, torch {cfm['meta']['torch']}, fp32")
    emit(f"- CFM: `{cfm['meta']['sampler']}`, block_size "
         f"{cfm['meta']['arch']['block_size']}, {cfm['meta']['n_params']:,} params, "
         f"step {cfm['meta']['checkpoint_step']}")
    cache_states = {r["kv_cache"] for r in bd3["rows"]}
    cache_note = ("KV cache on" if cache_states == {True}
                  else "KV cache OFF" if cache_states == {False} else "mixed KV cache")
    emit(f"- BD3-LM: block_size {bd3['meta']['block_size']}, "
         f"{bd3['meta']['n_params']:,} params, {bd3['meta']['weights_used_for_inference']} "
         f"weights, **{cache_note}**")
    emit("- CFM has no KV cache in any mode: `block_causal_sample` rebuilds the "
         "full `[clean; noisy]` 2L=512 forward on every flow-map jump.")
    emit(f"- Every number is the median of {cfm['meta']['repeats']} timed runs "
         f"after {cfm['meta']['warmup']} warmups, CUDA-synchronised.")
    emit()

    batches = sorted({r["batch_size"] for r in cfm["rows"]}
                     & {r["batch_size"] for r in bd3["rows"]})

    # ---------------------------------------------------------------- raw tables
    emit("## CFM")
    emit()
    emit("| batch | NFE (steps/block) | forwards/seq | s/batch | ms/seq | tok/s |")
    emit("|---|---|---|---|---|---|")
    for r in cfm["rows"]:
        emit(f"| {r['batch_size']} | {r['nfe']} | {r['forwards_per_sequence']} | "
             f"{r['latency_s_per_batch']:.4f} | {r['latency_ms_per_sequence']:.1f} | "
             f"{r['tokens_per_sec']:.0f} |")
    emit()

    emit("## BD3-LM")
    emit()
    emit("| batch | sampler | steps/block | NFE/seq | s/batch | ms/seq | tok/s |")
    emit("|---|---|---|---|---|---|---|")
    for r in bd3["rows"]:
        emit(f"| {r['batch_size']} | {r['sampler']} | {r['steps_per_block']} | "
             f"{r['nfe_per_sequence']} | {r['latency_s_per_batch']:.4f} | "
             f"{r['latency_ms_per_sequence']:.1f} | {r['tokens_per_sec']:.0f} |")
    emit()

    # -------------------------------------------------------------- iso-latency
    emit("## Iso-latency: which BD3-LM `steps_per_block` matches CFM")
    emit()
    for batch in batches:
        anc = [r for r in bd3["rows"]
               if r["batch_size"] == batch and r["sampler"] == "ancestral"]
        if len(anc) < 2:
            continue
        alpha, beta = fit_affine([r["steps_per_block"] for r in anc],
                                 [r["latency_s_per_batch"] for r in anc])
        measured = {r["steps_per_block"]: r for r in anc}
        emit(f"### batch = {batch}")
        emit()
        emit(f"Fit: `T(S) = {alpha:.4f}*S + {beta:.4f}` s/batch "
             f"(alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).")
        emit()
        emit("| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |")
        emit("|---|---|---|---|---|---|")
        for r in sorted((x for x in cfm["rows"] if x["batch_size"] == batch),
                        key=lambda x: x["nfe"]):
            target = r["latency_s_per_batch"]
            exact = (target - beta) / alpha
            near = min(measured, key=lambda s: abs(s - exact)) if measured else None
            got = measured[near]
            ratio = target / got["latency_s_per_batch"]
            verdict = (f"{ratio:.2f}x faster" if ratio > 1 else f"{1/ratio:.2f}x slower")
            emit(f"| {r['nfe']} | {r['latency_ms_per_sequence']:.1f} | "
                 f"**{exact:.1f}** | {near} | {got['latency_ms_per_sequence']:.1f} | "
                 f"{verdict} at S={near} |")
        emit()

    (RESULTS / args.out).write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {RESULTS / args.out}")


if __name__ == "__main__":
    main()
