"""Process 1 of the eval contract: a model in, a text dump out.

This is the *only* stage that touches a generative model. It samples texts at
one or more NFE settings and writes them to a self-describing dump; every
metric is then computed downstream by ``eval/score_texts.py``, which never sees
a checkpoint. Adding a new metric therefore never requires re-sampling.

Architecture is auto-detected from the checkpoint (vocab/hidden/layers/heads/
cond_dim/length/embedding type from tensor shapes, ``block_size`` from the saved
hyper-parameters), so the same command works for every model in this repo.

Real data is dumped in the identical schema via ``--gold``, so the reference
line flows through exactly the same metric code as the models.

    python -m eval.dump_samples --checkpoint path/to.ckpt \
        --nfe 1 2 4 8 16 --discretize argmax sample --n-samples 512 \
        --out dumps/ts_m3_step70000.json

    python -m eval.dump_samples --gold tinystories --n-samples 2048 \
        --out dumps/gold_tinystories.json

Output: ``<out>.json`` (provenance + texts) and ``<out>.tokens.npz`` (token ids
per point, so entropy is computed on the exact ids the model emitted rather
than on a re-tokenization).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from block.block_dit import BlockDIT  # noqa: E402
from block.block_semicat import BlockSemicatModule  # noqa: E402
from block.sampling import block_causal_sample, blockwise_sample  # noqa: E402
from semicat.models.semicat import SemicatModule  # noqa: E402
from semicat.net.duo import DIT  # noqa: E402

SCHEMA = "bcfm-textdump/v1"


# --------------------------------------------------------------------- helpers
def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except Exception:
        return None


def _sha256(path: Path, limit: int = 64 << 20) -> str:
    """Hash the first ``limit`` bytes — enough to identify a checkpoint cheaply."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _net_state(raw: dict) -> dict:
    state = {}
    for key, value in raw.items():
        if not key.startswith("net."):
            continue
        rest = key[len("net."):]
        if rest.startswith("_orig_mod."):  # saved under torch.compile
            rest = rest[len("_orig_mod."):]
        state[rest] = value
    return state


def detect_arch(checkpoint: dict) -> dict:
    """Infer the network shape from tensors, so no per-model config is needed."""
    state = _net_state(checkpoint["state_dict"])
    vocab_size, hidden_size = state["output_layer.linear.weight"].shape
    rotary = state["rotary_emb.cos_cached"].shape        # (1, L, 3, 1, head_dim)
    length, head_dim = rotary[1], rotary[-1]
    n_blocks = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    cond_dim = state["s_map.mlp.0.weight"].shape[0]
    embed_type = "rms" if any("final_norm" in k for k in state) else "naive"
    return {
        "vocab_size": int(vocab_size), "hidden_size": int(hidden_size),
        "length": int(length), "n_heads": int(hidden_size // head_dim),
        "n_blocks": int(n_blocks), "cond_dim": int(cond_dim),
        "embed_type": embed_type,
        "block_size": (checkpoint.get("hyper_parameters") or {}).get("block_size"),
    }


def build_module(arch: dict, device: str):
    """Full-sequence ``SemicatModule`` or block ``BlockSemicatModule``, per arch."""
    common = dict(vocab_size=arch["vocab_size"], hidden_size=arch["hidden_size"],
                  cond_dim=arch["cond_dim"], n_blocks=arch["n_blocks"],
                  n_heads=arch["n_heads"], dropout=0.1, length=arch["length"],
                  embed_type=arch["embed_type"])
    in_shape = (arch["length"], arch["vocab_size"])
    optimizer = functools.partial(torch.optim.AdamW, lr=1e-4)
    if arch["block_size"]:
        net = BlockDIT(block_size=arch["block_size"], attention_backend="flex",
                       jvp_attention_backend="auto", flex_kernel_block_size=64,
                       **common)
        module = BlockSemicatModule(
            net=net, optimizer=optimizer, scheduler=None, in_shape=in_shape,
            prior_type="gaussian", sd_type="lag", sd_prop=0.25,
            block_size=arch["block_size"], time_eps=0.0)
    else:
        net = DIT(**common)
        module = SemicatModule(net=net, optimizer=optimizer, scheduler=None,
                               in_shape=in_shape, prior_type="gaussian",
                               sd_type="lag")
    module.net.load_state_dict(_net_state(torch.load(
        str(arch["_path"]), map_location="cpu", weights_only=False)["state_dict"]),
        strict=True)
    return module.eval().to(device)


def make_detokenizer(vocab_size: int, data_dir: str):
    """text8 uses its character table; anything at GPT-2 vocab uses GPT-2."""
    if vocab_size == 27:
        meta = pickle.load(open(os.path.join(data_dir, "text8", "meta.pkl"), "rb"))
        itos = meta["itos"]
        return ("text8-chars",
                lambda ids: ["".join(itos[i] for i in row) for row in ids.tolist()])
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    return ("gpt2", lambda ids: tok.batch_decode(ids, skip_special_tokens=True))


def _discretize(endpoint: torch.Tensor, how: str) -> torch.Tensor:
    if how == "argmax":
        return endpoint.argmax(dim=-1)
    probs = endpoint.clamp_min(0).flatten(0, 1)
    return torch.multinomial(probs, 1).squeeze(-1).view(endpoint.shape[:-1])


# --------------------------------------------------------------------- sampling
@torch.inference_mode()
def sample_point(module, kind, arch, nfe, discretize, n_samples, batch_size):
    out, remaining = [], n_samples
    while remaining > 0:
        size = min(batch_size, remaining)
        if kind == "full_cfm":
            endpoint = module.sample_flow_map_batch(size, nfe)
            ids = _discretize(endpoint, discretize)
        elif kind == "block_causal":
            ids = block_causal_sample(module, block_size=arch["block_size"],
                                      steps_per_block=nfe, batch_size=size,
                                      discretize=discretize)
        elif kind == "blockwise_infer":
            ids = blockwise_sample(module, block_size=arch["_infer_block_size"],
                                   steps_per_block=nfe, batch_size=size,
                                   discretize=discretize)
        else:
            raise ValueError(f"unknown sampler kind {kind!r}")
        out.append(ids.cpu())
        remaining -= size
    return torch.cat(out, dim=0)[:n_samples]


def gold_tokens(dataset: str, n: int, data_dir: str, cache_dir: str):
    """Real held-out data, dumped in the same schema as model samples."""
    if dataset == "text8":
        from semicat.data.text8 import Text8DataModule
        dm = Text8DataModule(train_val_test_split=(351563, 20000, 19063), k=256,
                             data_dir=os.path.join(data_dir, "text8"),
                             batch_size=128, num_workers=0,
                             split_units="sequences")
        dm.trainer = SimpleNamespace(
            strategy=SimpleNamespace(root_device=torch.device("cpu")), world_size=1)
        dm.setup("fit")
        source = dm.data_test
        pick = torch.linspace(0, len(source) - 1, n).round().long().tolist()
        return torch.stack([source[i] for i in pick]).cpu(), 27, "test"
    from semicat.data.tinystories import TinyStoriesDataModule
    dm = TinyStoriesDataModule(cache_dir=cache_dir, batch_size=32, max_length=256,
                               tokenizer_name="gpt2", num_workers=0, num_proc=1,
                               pin_memory=False)
    dm._load_tokenizer()
    source = dm._load_dataset("validation")
    pick = torch.linspace(0, len(source) - 1, n).round().long().tolist()
    ids = torch.tensor([source[i]["input_ids"] for i in pick], dtype=torch.long)
    return ids, 50257, "validation"


# ------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a model into a text dump.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--gold", choices=["text8", "tinystories"],
                        help="dump real held-out data instead of model samples")
    parser.add_argument("--sampler", default="auto",
                        choices=["auto", "full_cfm", "block_causal", "blockwise_infer"])
    parser.add_argument("--infer-block-size", type=int, default=16,
                        help="block size for blockwise_infer (M2) on a full-seq model")
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--discretize", nargs="+", default=["argmax"],
                        choices=["argmax", "sample"])
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="data/tinystories")
    parser.add_argument("--label", default=None, help="human name for this dump")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points, token_arrays = [], {}

    if args.gold:
        ids, vocab_size, split = gold_tokens(args.gold, args.n_samples,
                                             args.data_dir, args.cache_dir)
        name, detok = make_detokenizer(vocab_size, args.data_dir)
        points.append({"id": "gold", "nfe": None, "discretize": None,
                       "forwards_per_sequence": 0, "n_samples": int(ids.shape[0]),
                       "sampling_seconds": 0.0, "texts": detok(ids)})
        token_arrays["gold"] = ids.numpy().astype(np.int32)
        model_meta = {"kind": "gold", "dataset": args.gold, "split": split,
                      "tokenizer": name, "vocab_size": vocab_size}
        sampler_meta = {"kind": "gold", "length": int(ids.shape[1]), "seed": None}
        print(f"[gold] {args.gold} {split}: {ids.shape[0]} texts", flush=True)
    else:
        path = Path(args.checkpoint)
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        arch = detect_arch(checkpoint)
        arch["_path"] = path
        arch["_infer_block_size"] = args.infer_block_size
        step = checkpoint.get("global_step")
        del checkpoint

        kind = args.sampler
        if kind == "auto":
            kind = "block_causal" if arch["block_size"] else "full_cfm"
        if kind == "block_causal" and not arch["block_size"]:
            # This is M2 as the paper defines it (main.tex 139/141): the training-free
            # variant "reuses a pretrained full-sequence CFM as the head ... running the
            # same blockwise sampler", and "both variants share one loop". So a
            # checkpoint with no trained block_size is run through BlockDIT at
            # --infer-block-size, and M2 differs from M3 only in which weights are
            # loaded. BlockDIT shares duo.DIT's trainable namespace, so the strict load
            # in build_module still succeeds.
            #
            # `--sampler blockwise_infer` is a *different* algorithm (single stream, the
            # prefix pinned over the noise, one scalar time, no mask). Comparing it
            # against M3 measures sampler + training together, not training alone.
            arch["block_size"] = args.infer_block_size

        module = build_module(arch, args.device)
        vocab_size = arch["vocab_size"]
        tok_name, detok = make_detokenizer(vocab_size, args.data_dir)
        dataset = "text8" if vocab_size == 27 else "tinystories"
        block_size = arch["block_size"] or (
            args.infer_block_size if kind == "blockwise_infer" else None)
        per_seq = (arch["length"] // block_size) if block_size else 1
        print(f"[model] {path.name} step={step} {dataset} {kind} "
              f"arch={arch['hidden_size']}x{arch['n_blocks']} vocab={vocab_size} "
              f"block_size={block_size}", flush=True)

        for discretize in args.discretize:
            for nfe in args.nfe:
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                started = time.time()
                ids = sample_point(module, kind, arch, nfe, discretize,
                                   args.n_samples, args.batch_size)
                elapsed = time.time() - started
                point_id = f"{discretize}_nfe{nfe}"
                points.append({
                    "id": point_id, "nfe": nfe, "discretize": discretize,
                    "forwards_per_sequence": nfe * per_seq,
                    "n_samples": int(ids.shape[0]),
                    "sampling_seconds": round(elapsed, 1),
                    "texts": detok(ids)})
                token_arrays[point_id] = ids.numpy().astype(np.int32)
                print(f"  [{point_id}] {ids.shape[0]} texts in {elapsed:.0f}s "
                      f"({nfe * per_seq} fwd/seq)", flush=True)
                torch.cuda.empty_cache()
                _write(out_path, SCHEMA, model_meta_from(path, step, arch, dataset,
                                                         tok_name),
                       {"kind": kind, "block_size": block_size,
                        "length": arch["length"], "seed": args.seed},
                       points, token_arrays, args.label)
        return _finish(out_path)

    _write(out_path, SCHEMA, model_meta, sampler_meta, points, token_arrays,
           args.label)
    _finish(out_path)


def model_meta_from(path: Path, step, arch: dict, dataset: str, tok_name: str) -> dict:
    clean = {k: v for k, v in arch.items() if not k.startswith("_")}
    return {"kind": "checkpoint", "checkpoint": str(path),
            "checkpoint_step": step, "checkpoint_sha256_head": _sha256(path),
            "dataset": dataset, "tokenizer": tok_name, "arch": clean}


def _write(out_path, schema, model_meta, sampler_meta, points, tokens, label):
    payload = {"schema": schema, "label": label or out_path.stem,
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git_commit": _git_commit(), "model": model_meta,
               "sampler": sampler_meta, "points": points}
    out_path.write_text(json.dumps(payload, indent=1))
    np.savez_compressed(out_path.with_suffix(".tokens.npz"), **tokens)


def _finish(out_path):
    size = out_path.stat().st_size / 1e6
    print(f"[done] {out_path} ({size:.1f} MB) + {out_path.with_suffix('.tokens.npz').name}",
          flush=True)


if __name__ == "__main__":
    main()
