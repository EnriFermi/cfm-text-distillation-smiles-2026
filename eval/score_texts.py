"""Process 2 of the eval contract: text dumps in, metrics out.

This stage never loads a generative model or a checkpoint. Its only inputs are
dumps produced by ``eval/dump_samples.py`` (and, for comparative metrics, a
reference dump of real data in that same schema). Adding a metric here is
therefore free: no re-sampling, no GPU time spent regenerating text.

Some metrics do load an *external* model — a judge LM for gen-PPL, GPT-2 for
MAUVE's featurizer — but never the model under test, which is the point.

    python -m eval.score_texts --dumps dumps/*.json \
        --reference dumps/gold_tinystories.json \
        --metrics entropy rep diversity ngram mauve \
        --out results/scores.json

Metric groups:
  entropy    mean token entropy (nats), on the exact ids the model emitted
  rep        seq-rep-n and distinct-n: catches degenerate repetition
  diversity  cross-sample n-gram overlap: catches mode collapse
  ngram      Jensen-Shannon divergence of uni/bigram stats against the reference
  mauve      distribution distance in GPT-2 embedding space (needs reference)
  gen_ppl    perplexity under a judge LM (--judge), tokenized by that judge
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "bcfm-textdump/v1"


# ----------------------------------------------------------------- dump loading
def load_dump(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, got {payload.get('schema')}")
    tokens_path = Path(path).with_suffix(".tokens.npz")
    payload["_tokens"] = dict(np.load(tokens_path)) if tokens_path.exists() else {}
    payload["_path"] = path
    return payload


def point_units(dump: dict, point: dict) -> list[list]:
    """Token ids when the dump carries them, else whitespace words."""
    array = dump["_tokens"].get(point["id"])
    if array is not None:
        return [list(row) for row in array]
    return [text.split() for text in point["texts"]]


# ----------------------------------------------------------- reference-free bits
def mean_token_entropy(sequences: list[list]) -> float:
    """Mean over sequences of the within-sequence unit entropy, in nats."""
    values = []
    for seq in sequences:
        counts = np.array(list(Counter(seq).values()), dtype=np.float64)
        probs = counts / counts.sum()
        values.append(float(-(probs * np.log(probs)).sum()))
    return float(np.mean(values))


def _ngrams(seq: list, n: int):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def repetition(sequences: list[list], orders=(2, 3, 4)) -> dict:
    """seq-rep-n (within-sequence repeats) and distinct-n (corpus-level variety)."""
    out = {}
    for n in orders:
        reps, uniq_all, total_all = [], set(), 0
        for seq in sequences:
            grams = _ngrams(seq, n)
            if not grams:
                continue
            reps.append(1.0 - len(set(grams)) / len(grams))
            uniq_all.update(grams)
            total_all += len(grams)
        out[f"seq_rep_{n}"] = float(np.mean(reps)) if reps else None
        out[f"distinct_{n}"] = (len(uniq_all) / total_all) if total_all else None
    return out


def cross_sample_overlap(sequences: list[list], n: int = 4, limit: int = 256) -> float:
    """Mean Jaccard overlap of n-gram sets between distinct samples (mode collapse)."""
    sets = [set(_ngrams(seq, n)) for seq in sequences[:limit]]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return float("nan")
    scores = []
    for i in range(0, len(sets) - 1, 2):          # disjoint pairs: O(n), unbiased
        a, b = sets[i], sets[i + 1]
        scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores))


def zipf_coefficient(sequences: list[list]) -> float:
    """Slope of log-frequency vs log-rank; natural text sits near 1."""
    counts = Counter()
    for seq in sequences:
        counts.update(seq)
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)
    if len(freqs) < 10:
        return float("nan")
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    slope, _ = np.polyfit(np.log(ranks), np.log(freqs), 1)
    return float(-slope)


# ------------------------------------------------------------ reference-relative
def _distribution(sequences: list[list], n: int) -> Counter:
    counts = Counter()
    for seq in sequences:
        counts.update(_ngrams(seq, n))
    return counts


def js_divergence(counts_p: Counter, counts_q: Counter) -> float:
    """Jensen-Shannon divergence in nats; 0 = identical, ln2 = disjoint."""
    total_p, total_q = sum(counts_p.values()), sum(counts_q.values())
    if not total_p or not total_q:
        return float("nan")
    divergence = 0.0
    for key in set(counts_p) | set(counts_q):
        p = counts_p.get(key, 0) / total_p
        q = counts_q.get(key, 0) / total_q
        m = 0.5 * (p + q)
        if p:
            divergence += 0.5 * p * math.log(p / m)
        if q:
            divergence += 0.5 * q * math.log(q / m)
    return float(divergence)


# --------------------------------------------------------------- external models
def compute_mauve(
    reference_texts,
    texts,
    featurize_model,
    device_id,
    max_length,
    reference_feature_cache=None,
):
    import mauve
    from mauve.utils import featurize_tokens_from_model, get_model, get_tokenizer
    tok = get_tokenizer(featurize_model)
    model = get_model(featurize_model, tok, device_id)

    def feats(items, name):
        encoded = [tok.encode(t, return_tensors="pt", truncation=True,
                              max_length=max_length) for t in items]
        return featurize_tokens_from_model(
            model, encoded, 16, name=name, verbose=True
        ).numpy()

    reference_features = None
    cache_path = Path(reference_feature_cache) if reference_feature_cache else None
    digest = hashlib.sha256()
    for item in reference_texts:
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    expected_metadata = {
        "schema": "mauve-reference-features/v1",
        "featurize_model": featurize_model,
        "max_length": int(max_length),
        "n_texts": len(reference_texts),
        "texts_sha256": digest.hexdigest(),
    }
    if cache_path and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(cached["metadata"].item()))
        if metadata != expected_metadata:
            raise ValueError(
                f"stale/incompatible MAUVE reference cache {cache_path}: "
                f"expected {expected_metadata}, got {metadata}"
            )
        reference_features = cached["features"]
        print(
            f"[mauve] reference cache hit: {cache_path} "
            f"shape={reference_features.shape}",
            flush=True,
        )
    else:
        print(
            f"[mauve] reference cache miss: featurizing {len(reference_texts)} texts",
            flush=True,
        )
        reference_features = feats(reference_texts, "reference")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez_compressed(
                temporary,
                features=reference_features,
                metadata=np.asarray(json.dumps(expected_metadata, sort_keys=True)),
            )
            temporary.replace(cache_path)
            print(f"[mauve] reference cache written: {cache_path}", flush=True)

    generated_features = feats(texts, "generated")
    del model, tok
    gc.collect()
    if device_id >= 0:
        import torch
        torch.cuda.empty_cache()
    result = mauve.compute_mauve(p_features=reference_features,
                                 q_features=generated_features,
                                 num_buckets="auto", seed=25, verbose=False)
    return float(result.mauve), float(result.frontier_integral)


def compute_gen_ppl(texts, judge, batch_size, context_size, device, dtype):
    """Judge-tokenized perplexity; mask-after-first-eos, as upstream does."""
    import torch
    import torch.nn.functional as F
    import transformers
    print(
        f"[gen_ppl] stage=judge_load model={judge} device={device} dtype={dtype}",
        flush=True,
    )
    tok = transformers.AutoTokenizer.from_pretrained(judge)
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    tok.padding_side, tok.truncation_side = "right", "right"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        judge, torch_dtype=getattr(torch, dtype)).eval().to(device)
    print(
        f"[gen_ppl] stage=judge_ready model={judge} "
        f"parameter_dtype={next(model.parameters()).dtype}",
        flush=True,
    )
    encoded = tok(texts, return_tensors="pt", return_attention_mask=True,
                  truncation=True, padding=True, max_length=context_size,
                  add_special_tokens=False)
    ids, attn = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
    total_nll, total_tokens = 0.0, 0
    n_batches = math.ceil(ids.size(0) / batch_size)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, ids.size(0), batch_size)):
            chunk, mask = ids[start:start + batch_size], attn[start:start + batch_size]
            logits = model(chunk, attention_mask=mask).logits.transpose(-1, -2)
            nll = F.cross_entropy(logits[..., :-1].float(), chunk[..., 1:],
                                  reduction="none")
            first_eos = (chunk == tok.eos_token_id).cumsum(-1) == 1
            valid = (first_eos[..., 1:] + (chunk != tok.eos_token_id)[..., 1:]).float()
            total_nll += (nll * valid).sum().item()
            total_tokens += valid.sum().item()
            if (
                batch_index == 0
                or (batch_index + 1) % 8 == 0
                or batch_index + 1 == n_batches
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"[gen_ppl] stage=judge batch={batch_index + 1}/{n_batches} "
                    f"samples={min(start + batch_size, ids.size(0))}/{ids.size(0)} "
                    f"tokens={total_tokens} elapsed={elapsed:.1f}s "
                    f"batches_per_s={(batch_index + 1) / max(elapsed, 1e-9):.2f}",
                    flush=True,
                )
    if total_tokens == 0:
        raise ValueError("gen-PPL has zero valid judge tokens")
    del model, tok, encoded, ids, attn
    gc.collect()
    torch.cuda.empty_cache()
    return float(np.exp(total_nll / total_tokens))


# ------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Score text dumps. No model needed.")
    parser.add_argument("--dumps", nargs="+", required=True)
    parser.add_argument("--reference", help="gold dump, for mauve/ngram")
    parser.add_argument("--metrics", nargs="+",
                        default=["entropy", "rep", "diversity", "ngram"],
                        choices=["entropy", "rep", "diversity", "zipf", "ngram",
                                 "mauve", "gen_ppl"])
    parser.add_argument("--judge", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-dtype", default="float32")
    parser.add_argument("--featurize-model", default="gpt2-large")
    parser.add_argument(
        "--mauve-reference-cache",
        help="validated .npz cache for reference GPT features",
    )
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted({p for pattern in args.dumps for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no dumps matched")

    reference, ref_units, ref_texts = None, None, None
    if args.reference:
        reference = load_dump(args.reference)
        ref_point = reference["points"][0]
        ref_units = point_units(reference, ref_point)
        ref_texts = ref_point["texts"]
        print(f"[reference] {args.reference}: {len(ref_texts)} texts", flush=True)

    needs_reference = {"mauve", "ngram"} & set(args.metrics)
    if needs_reference and reference is None:
        raise SystemExit(f"{sorted(needs_reference)} need --reference")

    rows = []
    for path in paths:
        dump = load_dump(path)
        label = dump.get("label") or Path(path).stem
        for point in dump["points"]:
            units = point_units(dump, point)
            row = {"dump": path, "label": label,
                   "dataset": dump["model"].get("dataset"),
                   "checkpoint_step": dump["model"].get("checkpoint_step"),
                   "sampler": dump["sampler"].get("kind"),
                   "block_size": dump["sampler"].get("block_size"),
                   "point": point["id"], "nfe": point["nfe"],
                   "discretize": point["discretize"],
                   "forwards_per_sequence": point["forwards_per_sequence"],
                   "n_samples": point["n_samples"]}
            started = time.time()
            if "entropy" in args.metrics:
                row["token_entropy"] = mean_token_entropy(units)
            if "rep" in args.metrics:
                row.update(repetition(units))
            if "diversity" in args.metrics:
                row["cross_sample_overlap_4"] = cross_sample_overlap(units)
            if "zipf" in args.metrics:
                row["zipf"] = zipf_coefficient(units)
            if "ngram" in args.metrics:
                for n in (1, 2):
                    row[f"js_{n}gram"] = js_divergence(_distribution(ref_units, n),
                                                       _distribution(units, n))
            if "mauve" in args.metrics:
                score, frontier = compute_mauve(ref_texts, point["texts"],
                                                args.featurize_model, args.device_id,
                                                args.context_size,
                                                args.mauve_reference_cache)
                row["mauve"], row["frontier_integral"] = score, frontier
            if "gen_ppl" in args.metrics:
                row["gen_ppl"] = compute_gen_ppl(point["texts"], args.judge,
                                                 args.judge_batch_size,
                                                 args.context_size, args.device,
                                                 args.judge_dtype)
                row["judge"] = args.judge
            row["scoring_seconds"] = round(time.time() - started, 1)
            rows.append(row)
            summary = " ".join(
                f"{k}={v:.4f}" for k, v in row.items()
                if isinstance(v, float) and k != "scoring_seconds")
            print(f"[{label}/{point['id']}] {summary}", flush=True)
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"metrics": args.metrics, "reference": args.reference,
                 "judge": args.judge if "gen_ppl" in args.metrics else None,
                 "rows": rows}, indent=1))
    print(f"[done] {args.out}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
