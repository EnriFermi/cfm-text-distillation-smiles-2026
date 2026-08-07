"""Generate TinyStories text over an NFE grid, argmax vs sampling discretization.

BCFM mode: ``block/sampling.py::block_causal_sample`` with B=16, so "NFE" here is
steps-per-block; total forwards per sequence = NFE x (256/16) = NFE x 16.

Full texts go to a .txt file; the console shows truncated previews. With
``discretize=argmax`` the per-block priors are identical across NFE for a fixed
seed, so the same latent is visibly refined as NFE grows; with ``sample`` the
multinomial draws consume RNG, so later blocks diverge by design.
"""
import argparse, json, sys, time
import torch
import transformers

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from gen_ppl_bcfm_ts import build_module, load_weights  # noqa: E402
from block.sampling import block_causal_sample  # noqa: E402
from semicat.metric.text_dist import TextMetrics  # noqa: E402


@torch.inference_mode()
def sample_once(module, block_size, spb, n, discretize):
    return block_causal_sample(
        module, block_size=block_size, steps_per_block=spb,
        batch_size=n, discretize=discretize,
    ).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--nfe", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--tokenizer", default="gpt2-large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-txt", default="gen_text_grid_ts.txt")
    ap.add_argument("--out-json", default="gen_text_grid_ts.json")
    ap.add_argument("--preview", type=int, default=170)
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    module = build_module(args.block_size, args.device)
    info = load_weights(module, args.checkpoint, "finetuned")
    n_blocks = 256 // args.block_size
    print(f"[load] step={info['step']} tensors={info['tensors']} "
          f"block_size={args.block_size} blocks={n_blocks}", flush=True)

    records, handle = [], open(args.out_txt, "w")
    handle.write(f"checkpoint step {info['step']} | block_size {args.block_size} "
                 f"| seed {args.seed} | n={args.n_samples}\n")
    for discretize in ("argmax", "sample"):
        print(f"\n########## discretize = {discretize} ##########", flush=True)
        handle.write(f"\n{'#'*78}\n### discretize = {discretize}\n{'#'*78}\n")
        for spb in args.nfe:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            t0 = time.time()
            ids = sample_once(module, args.block_size, spb, args.n_samples, discretize)
            dt = time.time() - t0
            texts = tok.batch_decode(ids, skip_special_tokens=True)
            ent = TextMetrics.compute_mean_entropy([ids])
            head = (f"NFE={spb} (fwd/seq={spb*n_blocks})  entropy={ent:.4f}  "
                    f"{dt:.1f}s")
            print(f"\n--- {head} ---", flush=True)
            handle.write(f"\n{'='*78}\n{head}\n{'='*78}\n")
            for i, text in enumerate(texts):
                print(f"  [{i}] {text[:args.preview]!r}", flush=True)
                handle.write(f"\n[{i}] {text}\n")
            records.append({
                "discretize": discretize, "nfe": spb,
                "forwards_per_sequence": spb * n_blocks,
                "token_entropy": ent, "seconds": round(dt, 1),
                "texts": texts,
            })
            handle.flush()
            torch.cuda.empty_cache()
    handle.close()
    with open(args.out_json, "w") as fh:
        json.dump({"checkpoint": args.checkpoint, "checkpoint_step": info["step"],
                   "block_size": args.block_size, "seed": args.seed,
                   "n_samples": args.n_samples, "records": records}, fh, indent=2)
    print(f"\n[done] {args.out_txt} / {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
