"""Generative PPL in BCFM (block-causal) mode for TinyStories checkpoints.

Both the fine-tune's source checkpoint and a fine-tuned checkpoint are evaluated
through the *same* sampler (``block/sampling.py::block_causal_sample``) and the
*same* architecture (``BlockDIT``), so the only difference is the weights — the
comparison isolates what the block fine-tune bought.

Scoring is the upstream pipeline verbatim (``TextMetrics.compute_mean_gen_ppl``,
gpt2-large tokenizer, mask-after-eos, PPL = exp(sum NLL / valid tokens)) with the
judge swapped to GPT-J-6B, matching the earlier full-sequence measurements.
"""
import argparse, functools, json, time
import torch
import transformers

from block.block_dit import BlockDIT
from block.block_semicat import BlockSemicatModule
from block.checkpoint import initialize_blockdit_from_cfm
from block.sampling import block_causal_sample
from semicat.metric.text_dist import TextMetrics

VOCAB, LENGTH = 50257, 256


def build_module(block_size, device):
    net = BlockDIT(
        vocab_size=VOCAB, hidden_size=384, cond_dim=128, n_blocks=6, n_heads=6,
        dropout=0.1, length=LENGTH, block_size=block_size, embed_type="rms",
        attention_backend="flex", jvp_attention_backend="auto",
        flex_kernel_block_size=64,
    )
    module = BlockSemicatModule(
        net=net, optimizer=functools.partial(torch.optim.AdamW, lr=1e-4),
        scheduler=None, in_shape=(LENGTH, VOCAB), prior_type="gaussian",
        sd_type="lag", sd_prop=0.25, block_size=block_size, time_eps=0.0,
    )
    return module.to(device).eval()


def load_weights(module, path, kind):
    """kind='source' -> full-sequence CFM ckpt; kind='finetuned' -> BlockSemicat ckpt."""
    if kind == "source":
        rep = initialize_blockdit_from_cfm(
            module.net, path, expected_source_sd_type="lag"
        )
        return {"step": rep.global_step, "tensors": rep.loaded_tensors}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {}
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("net."):
            continue
        rest = k[len("net."):]
        if rest.startswith("_orig_mod."):
            rest = rest[len("_orig_mod."):]
        sd[rest] = v
    module.net.load_state_dict(sd, strict=True)
    return {"step": int(ckpt.get("global_step", -1)), "tensors": len(sd)}


@torch.inference_mode()
def sample_ids(module, block_size, steps_per_block, n_samples, batch_size, device):
    out, remaining = [], n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        ids = block_causal_sample(
            module, block_size=block_size, steps_per_block=steps_per_block,
            batch_size=b, discretize="argmax",
        )
        out.append(ids.cpu())
        remaining -= b
    return torch.cat(out, dim=0)[:n_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-ckpt", required=True)
    ap.add_argument("--finetuned-ckpt", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps-per-block", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--sample-batch-size", type=int, default=32)
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default="EleutherAI/gpt-j-6B")
    ap.add_argument("--tokenizer", default="gpt2-large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gen_ppl_bcfm_ts.json")
    ap.add_argument("--print-samples", type=int, default=2)
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    n_blocks = LENGTH // args.block_size
    results = []

    for kind, path in (("source", args.source_ckpt), ("finetuned", args.finetuned_ckpt)):
        module = build_module(args.block_size, args.device)
        info = load_weights(module, path, kind)
        print(f"\n[{kind}] {path}\n  step={info['step']} tensors={info['tensors']}", flush=True)
        for spb in args.steps_per_block:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            t0 = time.time()
            ids = sample_ids(module, args.block_size, spb, args.n_samples,
                             args.sample_batch_size, args.device)
            strings = tok.batch_decode(ids, skip_special_tokens=True)
            entropy = TextMetrics.compute_mean_entropy([ids])
            t_sample = time.time() - t0
            t1 = time.time()
            ppl = TextMetrics.compute_mean_gen_ppl(
                strings, batch_size=args.judge_batch_size, context_size=LENGTH,
                ppl_model=args.judge_model, device=args.device,
            )
            t_judge = time.time() - t1
            row = {
                "model": kind, "checkpoint_step": info["step"],
                "block_size": args.block_size, "steps_per_block": spb,
                "forwards_per_sequence": spb * n_blocks,
                "gen_ppl": ppl, "nll": float(torch.log(torch.tensor(ppl))),
                "token_entropy": entropy,
                "sample_s": round(t_sample, 1), "judge_s": round(t_judge, 1),
            }
            results.append(row)
            print(f"  [{kind} spb={spb}] gen_ppl={ppl:.3f} nll={row['nll']:.4f} "
                  f"entropy={entropy:.4f} fwd/seq={row['forwards_per_sequence']} "
                  f"sample={t_sample:.0f}s judge={t_judge:.0f}s", flush=True)
            for i in range(min(args.print_samples, len(strings))):
                print(f"      [{i}] {strings[i][:180]!r}", flush=True)
            torch.cuda.empty_cache()
        del module
        torch.cuda.empty_cache()

    payload = {
        "mode": "bcfm_block_causal", "sampler": "block/sampling.py::block_causal_sample",
        "source_ckpt": args.source_ckpt, "finetuned_ckpt": args.finetuned_ckpt,
        "block_size": args.block_size, "n_samples": args.n_samples,
        "length": LENGTH, "seed": args.seed, "judge_model": args.judge_model,
        "judge_tokenizer": args.tokenizer, "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
