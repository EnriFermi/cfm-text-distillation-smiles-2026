"""BCFM-mode gen-PPL across the text8 M3 finetune's checkpoint series.

For every checkpoint of the 2026-07-18 block finetune, sample in block-causal
mode (``block/sampling.py::block_causal_sample``, B=16) at several
steps-per-block settings and score with the GPT-J-6B judge through the upstream
pipeline (``TextMetrics.compute_mean_gen_ppl``, gpt2-large tokenizer,
mask-after-eos). Detokenization uses the text8 char table, matching the earlier
text8 measurements.

Results are appended to the output JSON after every point, so a partial run is
still usable.
"""
import argparse, functools, json, pickle, time
from pathlib import Path

import torch

from block.block_dit import BlockDIT
from block.block_semicat import BlockSemicatModule
from block.sampling import block_causal_sample
from semicat.metric.text_dist import TextMetrics

VOCAB, LENGTH = 27, 256


def load_itos(meta_path):
    with open(meta_path, "rb") as handle:
        return pickle.load(handle)["itos"]


def tensor_to_strings(ids, itos):
    return ["".join(itos[i] for i in row) for row in ids.tolist()]


def build_module(block_size, device):
    net = BlockDIT(
        vocab_size=VOCAB, hidden_size=768, cond_dim=128, n_blocks=12, n_heads=12,
        dropout=0.1, length=LENGTH, block_size=block_size, embed_type="naive",
        attention_backend="flex", jvp_attention_backend="auto",
        flex_kernel_block_size=64,
    )
    module = BlockSemicatModule(
        net=net, optimizer=functools.partial(torch.optim.AdamW, lr=1e-4),
        scheduler=None, in_shape=(LENGTH, VOCAB), prior_type="gaussian",
        sd_type="lag", sd_prop=0.25, block_size=block_size, time_eps=0.0,
    )
    return module.to(device).eval()


def load_net(module, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("net."):
            continue
        rest = key[len("net."):]
        if rest.startswith("_orig_mod."):
            rest = rest[len("_orig_mod."):]
        sd[rest] = value
    module.net.load_state_dict(sd, strict=True)
    return int(ckpt.get("global_step", -1))


@torch.inference_mode()
def sample_ids(module, block_size, steps_per_block, n_samples, batch_size):
    out, remaining = [], n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out.append(
            block_causal_sample(
                module, block_size=block_size, steps_per_block=steps_per_block,
                batch_size=b, discretize="argmax",
            ).cpu()
        )
        remaining -= b
    return torch.cat(out, dim=0)[:n_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--meta", default="data/text8/meta.pkl")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps-per-block", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--sample-batch-size", type=int, default=128)
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default="EleutherAI/gpt-j-6B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gen_ppl_bcfm_text8_sweep.json")
    args = ap.parse_args()

    itos = load_itos(args.meta)
    n_blocks = LENGTH // args.block_size
    # step_XXXXXXX.ckpt only; last.ckpt duplicates the final one
    paths = sorted(Path(args.ckpt_dir).glob("step_*.ckpt"))
    print(f"[plan] {len(paths)} checkpoints x {len(args.steps_per_block)} settings "
          f"= {len(paths)*len(args.steps_per_block)} points", flush=True)

    module = build_module(args.block_size, args.device)
    results = []
    for path in paths:
        step = load_net(module, str(path))
        print(f"\n[ckpt] {path.name} step={step}", flush=True)
        for spb in args.steps_per_block:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            t0 = time.time()
            ids = sample_ids(module, args.block_size, spb, args.n_samples,
                             args.sample_batch_size)
            strings = tensor_to_strings(ids, itos)
            entropy = TextMetrics.compute_mean_entropy([ids])
            t_sample = time.time() - t0
            t1 = time.time()
            ppl = TextMetrics.compute_mean_gen_ppl(
                strings, batch_size=args.judge_batch_size, context_size=LENGTH,
                ppl_model=args.judge_model, device=args.device,
            )
            t_judge = time.time() - t1
            row = {
                "checkpoint": path.name, "step": step,
                "block_size": args.block_size, "steps_per_block": spb,
                "forwards_per_sequence": spb * n_blocks,
                "gen_ppl": ppl, "nll": float(torch.log(torch.tensor(ppl))),
                "char_entropy": entropy,
                "sample_s": round(t_sample, 1), "judge_s": round(t_judge, 1),
            }
            results.append(row)
            print(f"  [step={step} spb={spb}] gen_ppl={ppl:.3f} "
                  f"entropy={entropy:.4f} sample={t_sample:.0f}s judge={t_judge:.0f}s",
                  flush=True)
            # incremental write so a partial sweep is still usable
            with open(args.out, "w") as handle:
                json.dump({
                    "mode": "bcfm_block_causal",
                    "sampler": "block/sampling.py::block_causal_sample",
                    "ckpt_dir": args.ckpt_dir, "block_size": args.block_size,
                    "n_samples": args.n_samples, "length": LENGTH,
                    "seed": args.seed, "judge_model": args.judge_model,
                    "judge_tokenizer": "gpt2-large", "results": results,
                }, handle, indent=2)
            torch.cuda.empty_cache()
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
