"""Re-judge already-generated texts with a stronger LM. No sampling.

Reads the texts saved by ``mauve_grid_ts.py`` (reference + one list per grid
point) and scores them with an arbitrary causal LM.

Why not reuse ``TextMetrics.compute_mean_gen_ppl``: it calls
``_load_tokenizer()`` with no argument, so it always tokenizes with gpt2-large.
That happens to be correct for GPT-J-6B (same GPT-2 BPE vocab) but is wrong for
any other judge. Everything else here — truncation to ``context_size``, the
mask-after-first-eos rule, and ``PPL = exp(sum NLL / valid tokens)`` — follows
that function exactly, so numbers stay methodologically comparable within a judge.

Cross-judge PPL values are NOT comparable: a stronger judge assigns lower
perplexity to everything, which is why the reference is rescored here too and
the ratio to it is the meaningful quantity.
"""
import argparse, json, time
import torch
import torch.nn.functional as F
import transformers


def load_judge(name, device, dtype):
    tok = transformers.AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    tok.truncation_side = "right"
    # No low_cpu_mem_usage/device_map: those need `accelerate`, and plain loading
    # peaks at ~2x model size in CPU RAM, which this box has to spare.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype,
    ).eval().to(device)
    return tok, model


@torch.inference_mode()
def gen_ppl(texts, tok, model, batch_size, context_size, device):
    """Token-weighted PPL, mirroring upstream compute_mean_gen_ppl."""
    enc = tok(texts, return_tensors="pt", return_attention_mask=True,
              truncation=True, padding=True, max_length=context_size,
              add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    eos_id = tok.eos_token_id
    total_nll, total_tokens = 0.0, 0
    for start in range(0, ids.size(0), batch_size):
        chunk = ids[start:start + batch_size]
        mask = attn[start:start + batch_size]
        logits = model(chunk, attention_mask=mask).logits.transpose(-1, -2)
        nlls = F.cross_entropy(logits[..., :-1].float(), chunk[..., 1:],
                               reduction="none")
        # keep real tokens plus exactly the first eos, as upstream does
        first_eos = (chunk == eos_id).cumsum(-1) == 1
        token_mask = chunk != eos_id
        valid = (first_eos[..., 1:] + token_mask[..., 1:]).float()
        total_nll += (nlls * valid).sum().item()
        total_tokens += valid.sum().item()
    return float(torch.exp(torch.tensor(total_nll / total_tokens))), total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="JSON from mauve_grid_ts.py")
    ap.add_argument("--judge", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--context-size", type=int, default=256)
    ap.add_argument("--n-reference", type=int, default=512,
                    help="how many reference texts to score (they are the gold line)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="judge_llama.json")
    args = ap.parse_args()

    payload = json.load(open(args.texts))
    reference = payload["reference"][:args.n_reference]
    generated = payload["generated"]

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tok, model = load_judge(args.judge, args.device, dtype)
    print(f"[judge] {args.judge} | vocab {len(tok)} | loaded in {time.time()-t0:.0f}s",
          flush=True)

    results = []
    ppl, ntok = gen_ppl(reference, tok, model, args.batch_size,
                        args.context_size, args.device)
    print(f"[GOLD] real TinyStories: PPL={ppl:.4f}  ({len(reference)} texts, "
          f"{ntok:,} scored tokens)", flush=True)
    gold = ppl
    results.append({"point": "gold_reference", "n_texts": len(reference),
                    "gen_ppl": ppl, "ratio_to_gold": 1.0})

    for name, texts in generated.items():
        t1 = time.time()
        ppl, ntok = gen_ppl(texts, tok, model, args.batch_size,
                            args.context_size, args.device)
        row = {"point": name, "n_texts": len(texts), "gen_ppl": ppl,
               "ratio_to_gold": ppl / gold, "seconds": round(time.time() - t1, 1)}
        results.append(row)
        print(f"[{name}] PPL={ppl:.3f}  = {ppl/gold:.1f}x gold  "
              f"({time.time()-t1:.0f}s)", flush=True)
        with open(args.out, "w") as fh:
            json.dump({"judge": args.judge, "context_size": args.context_size,
                       "texts_file": args.texts, "gold_gen_ppl": gold,
                       "results": results}, fh, indent=2)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
