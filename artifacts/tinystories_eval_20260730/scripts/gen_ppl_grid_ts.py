"""GPT-J-6B gen-PPL over an NFE grid, argmax vs sampling, for one TS checkpoint.

BCFM mode (``block_causal_sample``, B=16), so NFE = steps-per-block and total
forwards per sequence = NFE x 16. Scoring is the upstream pipeline
(``TextMetrics.compute_mean_gen_ppl``, gpt2-large tokenizer, mask-after-eos)
with the GPT-J-6B judge, matching every earlier measurement.

Results are written after each point, so a partial run stays usable.
"""
import argparse, json, sys, time
import torch
import transformers

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from gen_ppl_bcfm_ts import build_module, load_weights  # noqa: E402
from block.sampling import block_causal_sample  # noqa: E402
from semicat.metric.text_dist import TextMetrics  # noqa: E402


@torch.inference_mode()
def sample_ids(module, block_size, spb, n_samples, batch_size, discretize):
    out, remaining = [], n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out.append(
            block_causal_sample(
                module, block_size=block_size, steps_per_block=spb,
                batch_size=b, discretize=discretize,
            ).cpu()
        )
        remaining -= b
    return torch.cat(out, dim=0)[:n_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--nfe", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--sample-batch-size", type=int, default=64)
    ap.add_argument("--judge-batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default="EleutherAI/gpt-j-6B")
    ap.add_argument("--tokenizer", default="gpt2-large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="gen_ppl_grid_ts.json")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    module = build_module(args.block_size, args.device)
    info = load_weights(module, args.checkpoint, "finetuned")
    n_blocks = 256 // args.block_size
    total = len(args.nfe) * 2
    print(f"[load] step={info['step']} tensors={info['tensors']} | {total} points",
          flush=True)

    results = []
    for discretize in ("argmax", "sample"):
        for spb in args.nfe:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            t0 = time.time()
            ids = sample_ids(module, args.block_size, spb, args.n_samples,
                             args.sample_batch_size, discretize)
            strings = tok.batch_decode(ids, skip_special_tokens=True)
            ent = TextMetrics.compute_mean_entropy([ids])
            t_sample = time.time() - t0
            t1 = time.time()
            ppl = TextMetrics.compute_mean_gen_ppl(
                strings, batch_size=args.judge_batch_size, context_size=256,
                ppl_model=args.judge_model, device=args.device,
            )
            t_judge = time.time() - t1
            row = {
                "discretize": discretize, "nfe": spb,
                "forwards_per_sequence": spb * n_blocks,
                "gen_ppl": ppl, "nll": float(torch.log(torch.tensor(ppl))),
                "token_entropy": ent,
                "sample_s": round(t_sample, 1), "judge_s": round(t_judge, 1),
            }
            results.append(row)
            print(f"[{discretize} nfe={spb}] gen_ppl={ppl:.3f} entropy={ent:.4f} "
                  f"fwd/seq={row['forwards_per_sequence']} "
                  f"sample={t_sample:.0f}s judge={t_judge:.0f}s", flush=True)
            with open(args.out, "w") as fh:
                json.dump({
                    "mode": "bcfm_block_causal", "checkpoint": args.checkpoint,
                    "checkpoint_step": info["step"], "block_size": args.block_size,
                    "n_samples": args.n_samples, "length": 256, "seed": args.seed,
                    "judge_model": args.judge_model, "judge_tokenizer": args.tokenizer,
                    "results": results,
                }, fh, indent=2)
            torch.cuda.empty_cache()
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
