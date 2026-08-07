"""MAUVE over the NFE x discretizer grid for a TinyStories BCFM checkpoint.

MAUVE compares the *distribution* of model text against real text in GPT-2-large
embedding space, penalising both degeneration (repetitive, under-diverse text)
and incoherence. That makes it the right complement to gen-PPL here, which we
found rewards repetition.

Reference (p) = real TinyStories validation windows, featurized once and reused.
Model (q) = block-causal samples at each (discretizer, NFE) point.
Generated texts are saved so later reference-free metrics (rep-n, n-gram
divergence) need no re-sampling.

Score is in [0, 1]; higher = model distribution closer to human text.
"""
import argparse, json, sys, time
import torch
import transformers

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import mauve  # noqa: E402
from mauve.utils import get_model, get_tokenizer, featurize_tokens_from_model  # noqa: E402
from gen_ppl_bcfm_ts import build_module, load_weights  # noqa: E402
from block.sampling import block_causal_sample  # noqa: E402
from semicat.data.tinystories import TinyStoriesDataModule  # noqa: E402
from semicat.metric.text_dist import TextMetrics  # noqa: E402

LENGTH = 256


def reference_texts(n, cache_dir, tokenizer):
    dm = TinyStoriesDataModule(cache_dir=cache_dir, batch_size=32, max_length=LENGTH,
                               tokenizer_name="gpt2", num_workers=0, num_proc=1,
                               pin_memory=False)
    dm._load_tokenizer()
    ds = dm._load_dataset("validation")
    idx = torch.linspace(0, len(ds) - 1, n).round().long().tolist()
    ids = torch.tensor([ds[i]["input_ids"] for i in idx], dtype=torch.long)
    return tokenizer.batch_decode(ids, skip_special_tokens=True), ids


@torch.inference_mode()
def sample_ids(module, block_size, spb, n_samples, batch_size, discretize):
    out, remaining = [], n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        out.append(block_causal_sample(
            module, block_size=block_size, steps_per_block=spb,
            batch_size=b, discretize=discretize).cpu())
        remaining -= b
    return torch.cat(out, dim=0)[:n_samples]


def featurize(texts, tok, model, batch_size):
    toks = [tok.encode(t, return_tensors="pt", truncation=True, max_length=LENGTH)
            for t in texts]
    return featurize_tokens_from_model(model, toks, batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", default="data/tinystories")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--n-samples", type=int, default=512)
    # More reference than generated: k-means is fit on the pooled p+q features, so
    # extra (free) real texts stabilise clustering. num_buckets uses min(|p|,|q|),
    # so this does not change the bucket count.
    ap.add_argument("--n-reference", type=int, default=2048)
    ap.add_argument("--sample-batch-size", type=int, default=64)
    ap.add_argument("--featurize-batch-size", type=int, default=16)
    ap.add_argument("--featurize-model", default="gpt2-large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--out", default="mauve_grid_ts.json")
    ap.add_argument("--texts-out", default="mauve_grid_texts.json")
    args = ap.parse_args()

    tok = get_tokenizer(args.featurize_model)
    feat_model = get_model(args.featurize_model, tok, args.device_id)
    print(f"[featurizer] {args.featurize_model} on device {args.device_id}", flush=True)

    p_text, p_ids = reference_texts(args.n_reference, args.cache_dir, tok)
    t0 = time.time()
    p_features = featurize(p_text, tok, feat_model, args.featurize_batch_size)
    ref_entropy = TextMetrics.compute_mean_entropy([p_ids])
    print(f"[reference] {len(p_text)} real TinyStories texts, entropy={ref_entropy:.4f}, "
          f"featurized in {time.time()-t0:.0f}s", flush=True)

    module = build_module(args.block_size, "cuda")
    info = load_weights(module, args.checkpoint, "finetuned")
    n_blocks = LENGTH // args.block_size
    print(f"[model] step={info['step']} | {len(args.nfe)*2} points", flush=True)

    results, all_texts = [], {}
    for discretize in ("argmax", "sample"):
        for spb in args.nfe:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            t0 = time.time()
            ids = sample_ids(module, args.block_size, spb, args.n_samples,
                             args.sample_batch_size, discretize)
            q_text = tok.batch_decode(ids, skip_special_tokens=True)
            ent = TextMetrics.compute_mean_entropy([ids])
            t_sample = time.time() - t0
            t1 = time.time()
            q_features = featurize(q_text, tok, feat_model, args.featurize_batch_size)
            out = mauve.compute_mauve(p_features=p_features, q_features=q_features,
                                      num_buckets="auto", seed=25, verbose=False)
            t_mauve = time.time() - t1
            row = {"discretize": discretize, "nfe": spb,
                   "forwards_per_sequence": spb * n_blocks,
                   "mauve": float(out.mauve),
                   "frontier_integral": float(out.frontier_integral),
                   "token_entropy": ent,
                   "sample_s": round(t_sample, 1), "mauve_s": round(t_mauve, 1)}
            results.append(row)
            all_texts[f"{discretize}_nfe{spb}"] = q_text
            print(f"[{discretize} nfe={spb}] MAUVE={out.mauve:.4f} "
                  f"FI={out.frontier_integral:.4f} entropy={ent:.4f} "
                  f"sample={t_sample:.0f}s mauve={t_mauve:.0f}s", flush=True)
            with open(args.out, "w") as fh:
                json.dump({"checkpoint": args.checkpoint,
                           "checkpoint_step": info["step"],
                           "block_size": args.block_size,
                           "n_samples": args.n_samples, "seed": args.seed,
                           "featurize_model": args.featurize_model,
                           "reference": "TinyStories validation",
                           "reference_entropy": ref_entropy,
                           "results": results}, fh, indent=2)
            with open(args.texts_out, "w") as fh:
                json.dump({"reference": p_text, "generated": all_texts}, fh)
            torch.cuda.empty_cache()
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
