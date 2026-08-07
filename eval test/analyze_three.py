"""Build the matched-NFE MDLM/CFM/BD3-LM speed report."""

from __future__ import annotations

import argparse
import json

from common import RESULTS


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm", default="cfm_latency_three_max.json")
    parser.add_argument("--bd3lm", default="bd3lm_latency_three_max.json")
    parser.add_argument("--mdlm", default="mdlm_latency_three_max.json")
    parser.add_argument("--out", default="comparison_three_max_20260807.md")
    args = parser.parse_args()

    payloads = {
        "CFM": load(args.cfm),
        "BD3-LM": load(args.bd3lm),
        "MDLM": load(args.mdlm),
    }
    cfm = {
        (row["batch_size"], row["nfe"]): row
        for row in payloads["CFM"]["rows"]
    }
    bd3 = {
        (row["batch_size"], row["steps_per_block"]): row
        for row in payloads["BD3-LM"]["rows"]
        if row.get("sampler") == "ancestral"
    }
    # L=256 and block=16 means one block-model step costs 16 denoiser calls.
    mdlm = {
        (row["batch_size"], row["num_steps"] // 16): row
        for row in payloads["MDLM"]["rows"]
        if row["num_steps"] % 16 == 0
    }
    common = sorted(set(cfm) & set(bd3) & set(mdlm))
    if not common:
        raise SystemExit("the three sweeps have no common (batch, matched-step) points")

    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    meta = payloads["MDLM"]["meta"]
    emit("# MDLM vs CFM vs BD3-LM — optimized generation speed")
    emit()
    emit(
        f"- GPU: **{meta['gpu']}**; PyTorch {meta['torch']}; sequence length 256."
    )
    emit(
        f"- Every point is the median of {meta['repeats']} synchronized runs "
        f"after {meta['warmup']} warmups."
    )
    emit(
        "- Matched denoiser budget: CFM/BD3-LM `S` steps per each of 16 blocks "
        "versus MDLM `16×S` full-sequence diffusion steps."
    )
    emit(
        "- CFM: sparse active-block vocab operations + finalized-prefix KV cache. "
        "BD3-LM: KV cache + lean generator + compiled dynamic backbone. "
        "MDLM: active-position vocab/sampling + lean generator + compiled static backbone."
    )
    dtypes = ", ".join(
        f"{name}={payload['meta']['dtype']}" for name, payload in payloads.items()
    )
    emit(f"- Precision: {dtypes}.")
    emit(
        "- MDLM active-position sampling is distribution-equivalent but consumes RNG "
        "differently; BF16 may also change individual samples."
    )
    emit()

    emit("## Latency per generated sequence")
    emit()
    emit("| batch | matched S | total scheduled NFE | CFM ms | BD3-LM ms | MDLM ms | winner |")
    emit("|---:|---:|---:|---:|---:|---:|:---|")
    for key in common:
        batch, steps = key
        rows = {"CFM": cfm[key], "BD3-LM": bd3[key], "MDLM": mdlm[key]}
        winner = min(rows, key=lambda name: rows[name]["latency_ms_per_sequence"])
        fastest = rows[winner]["latency_ms_per_sequence"]
        ordered = sorted(rows.items(), key=lambda item: item[1]["latency_ms_per_sequence"])
        runner_up = ordered[1][1]["latency_ms_per_sequence"]
        emit(
            f"| {batch} | {steps} | {16 * steps} | "
            f"{rows['CFM']['latency_ms_per_sequence']:.2f} | "
            f"{rows['BD3-LM']['latency_ms_per_sequence']:.2f} | "
            f"{rows['MDLM']['latency_ms_per_sequence']:.2f} | "
            f"**{winner}** ({runner_up / fastest:.2f}×) |"
        )
    emit()

    emit("## Best throughput in the measured grid")
    emit()
    emit("| model | tokens/s | batch | setting | peak allocated GB |")
    emit("|:---|---:|---:|:---|---:|")
    tables = {"CFM": cfm, "BD3-LM": bd3, "MDLM": mdlm}
    for name, table in tables.items():
        best = max(table.values(), key=lambda row: row["tokens_per_sec"])
        setting = (
            f"{best['num_steps']} total steps"
            if name == "MDLM"
            else f"{best.get('nfe', best.get('steps_per_block'))} steps/block"
        )
        emit(
            f"| {name} | **{best['tokens_per_sec']:,.1f}** | "
            f"{best['batch_size']} | {setting} | {best['peak_mem_gb']:.2f} |"
        )
    emit()

    emit("## Interpretation")
    emit()
    winners = []
    for key in common:
        rows = {"CFM": cfm[key], "BD3-LM": bd3[key], "MDLM": mdlm[key]}
        winners.append(min(rows, key=lambda name: rows[name]["latency_ms_per_sequence"]))
    counts = {name: winners.count(name) for name in tables}
    emit(
        "Across the common grid, "
        + ", ".join(f"{name} wins {counts[name]}/{len(common)} points" for name in tables)
        + "."
    )
    emit(
        "This is a speed comparison at matched scheduled denoiser calls, not a "
        "matched-quality comparison; generation quality must be evaluated separately."
    )

    path = RESULTS / args.out
    path.write_text("\n".join(lines) + "\n")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
