"""Regenerate fresh showcase samples for every evaluated (model, steps[, B]) config.

Writes ``results/generation_samples_text8_tinystories.txt`` with newly sampled
strings (not copied from old samples.txt).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from eval.generate import load_module, sample_tokens  # noqa: E402

N_SAMPLES = int(os.environ.get("SHOWCASE_N", "8"))
SEED = int(os.environ.get("SHOWCASE_SEED", "42"))
LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results" / "generation_samples_text8_tinystories.txt"

TEXT8_M1 = Path("/home/zolotovskijal/checkpoints/last.ckpt")
TEXT8_M3 = Path("/home/zolotovskijal/checkpoints/step_0090000.ckpt")
TS_M1 = Path("/home/zolotovskijal/baselines/NEW/tinystories_a100/checkpoints/best.ckpt")
BASE = Path("/home/zolotovskijal/baselines")
BASE_CODE = Path(os.environ.get("BASELINES_CODE", str(ROOT / "baselines")))
TS_DATA = Path(os.environ.get("TEXT8_DATA_DIR", str(ROOT / "data" / "tinystories")))


def _baselines_path() -> None:
    sys.path.insert(0, str(BASE_CODE))


def _text8_cfm_cfg(ckpt: Path) -> OmegaConf:
    return OmegaConf.create({
        "ckpt_path": str(ckpt),
        "model": {
            "_target_": "semicat.models.textsemicat.TextSemicatModule",
            "in_shape": [256, 27],
            "prior_type": "gaussian",
            "scheduler": None,
            "sd_type": "lag",
            "sd_prop": 0.25,
            "optimizer": None,
            "compile": False,
            "calc_nll": False,
            "net": {
                "_target_": "semicat.net.duo.DIT",
                "vocab_size": 27,
                "hidden_size": 768,
                "cond_dim": 128,
                "n_blocks": 12,
                "n_heads": 12,
                "dropout": 0.1,
                "length": 256,
                "embed_type": "naive",
            },
        },
    })


def _m3_cfg(ckpt: Path) -> OmegaConf:
    return OmegaConf.create({
        "ckpt_path": str(ckpt),
        "model": {
            "_target_": "block.block_semicat.BlockSemicatModule",
            "in_shape": [256, 27],
            "prior_type": "gaussian",
            "scheduler": None,
            "sd_type": "lag",
            "sd_prop": 0.25,
            "block_size": 16,
            "ecld_weight": 1.0,
            "time_eps": 0.0,
            "optimizer": None,
            "compile": False,
            "calc_nll": False,
            "net": {
                "_target_": "semicat.net.duo.DIT",
                "vocab_size": 27,
                "hidden_size": 768,
                "cond_dim": 128,
                "n_blocks": 12,
                "n_heads": 12,
                "dropout": 0.1,
                "length": 256,
                "embed_type": "naive",
            },
        },
    })


def _ts_cfm_cfg(ckpt: Path) -> OmegaConf:
    return OmegaConf.create({
        "ckpt_path": str(ckpt),
        "model": {
            "_target_": "semicat.models.textsemicat.TextSemicatModule",
            "in_shape": [256, 50257],
            "prior_type": "gaussian",
            "scheduler": None,
            "sd_type": "lag",
            "sd_prop": 0.25,
            "optimizer": None,
            "compile": False,
            "calc_nll": False,
            "net": {
                "_target_": "semicat.net.duo.DIT",
                "vocab_size": 50257,
                "hidden_size": 384,
                "cond_dim": 128,
                "n_blocks": 6,
                "n_heads": 6,
                "dropout": 0.1,
                "length": 256,
                "embed_type": "rms",
            },
        },
    })


def _decode_text8(tokens: torch.Tensor) -> list[str]:
    itos = {i: ch for i, ch in enumerate(" abcdefghijklmnopqrstuvwxyz")}
    return ["".join(itos.get(int(t), "?") for t in row) for row in tokens.tolist()]


def _decode_gpt2(tokens: torch.Tensor) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    return [tok.decode(row.tolist(), skip_special_tokens=True) for row in tokens]


def _gen_cfm(module, sampler_name: str, steps: int, block_size: int, batch: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if sampler_name == "full_cfm":
        sampler = OmegaConf.create({"name": "full_cfm", "steps": steps, "discretize": "argmax"})
    else:
        sampler = OmegaConf.create({
            "name": "bcfm_infer",
            "block_size": block_size,
            "steps_per_block": steps,
            "discretize": "argmax",
            "prefix": "generated",
            "schedule": None,
            "use_kv_cache": True,
        })
    return sample_tokens(module, sampler, n_samples=N_SAMPLES, batch_size=batch, length=LENGTH)


def _gen_baseline(ckpt: Path, *, num_steps=None, steps_per_block=None, first_hitting=False) -> list[str]:
    _baselines_path()
    from bcfm_baselines.checkpoint import load_checkpoint_model
    from bcfm_baselines.data import create_corpus
    from bcfm_baselines.models.generative_text import GenerationConfig

    model, payload = load_checkpoint_model(ckpt, DEVICE)
    config = payload["config"]
    # Point corpus at local data.
    name = config["dataset"]["name"]
    if name == "tinystories":
        config["dataset"]["data_dir"] = str(TS_DATA)
    else:
        config["dataset"]["data_dir"] = str(ROOT / "data" / "text8")
    corpus = create_corpus(config)
    sampling = dict(config["sampling"])
    sampling.update({"length": LENGTH, "batch_size": N_SAMPLES, "seed": SEED})
    if num_steps is not None:
        sampling["num_steps"] = num_steps
    if steps_per_block is not None:
        sampling["steps_per_block"] = steps_per_block
    sampling["first_hitting"] = first_hitting
    result = model.generate(GenerationConfig(**sampling))
    return corpus.decode(result.tokens)


def discover_configs() -> list[dict]:
    """One entry per (dataset, model, block_size, steps)."""
    best: dict[tuple, dict] = {}
    for p in Path(ROOT / "results").glob("*/metrics.json"):
        m = json.loads(p.read_text())
        if m.get("metric_schema") != "network_forwards_v2":
            continue
        if m.get("sampler") == "gold":
            continue
        ds = m.get("dataset") or (
            "tinystories" if str(m.get("model", "")).endswith("-TS") else "text8"
        )
        key = (ds, m["model"], int(m["block_size"]), int(m["steps_per_block"]))
        # Prefer seed0 metrics for ckpt path / flags.
        cand = {
            "dataset": ds,
            "model": m["model"],
            "block_size": int(m["block_size"]),
            "steps": int(m["steps_per_block"]),
            "sampler": m["sampler"],
            "first_hitting": bool(m.get("first_hitting") or False),
            "ckpt": m.get("ckpt"),
            "gen_ppl": m.get("gen_ppl"),
            "seed": int(m.get("seed", 0)),
            "exp": p.parent.name,
        }
        prev = best.get(key)
        if prev is None or (cand["seed"] == 0 and prev["seed"] != 0):
            best[key] = cand
        elif prev["seed"] != 0 and cand["gen_ppl"] is not None:
            # keep existing
            pass
    return sorted(best.values(), key=lambda c: (c["dataset"], c["model"], c["block_size"], c["steps"]))


def main() -> None:
    configs = discover_configs()
    print(f"configs={len(configs)} n_samples={N_SAMPLES} seed={SEED} device={DEVICE}")

    # Cache loaded CFM modules by (family, ckpt).
    cfm_cache: dict[str, object] = {}
    sections: list[str] = []
    sections.append("=" * 78)
    sections.append("Fresh generation samples: Text8 + TinyStories")
    sections.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    sections.append(f"n_samples={N_SAMPLES}  seed={SEED}  length={LENGTH}  (regenerated, not from old samples.txt)")
    sections.append("=" * 78)

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for c in configs:
        by_ds[c["dataset"]].append(c)

    for dataset in ("text8", "tinystories"):
        sections.append("")
        sections.append("#" * 78)
        sections.append(f"# DATASET: {dataset}")
        sections.append("#" * 78)

        for c in by_ds.get(dataset, []):
            model, steps, B = c["model"], c["steps"], c["block_size"]
            title = f"{model}, steps={steps}" + (f", B={B}" if model in ("M2", "M2-TS", "M3", "BD3LM", "BD3LM-TS") or (model.startswith("M2")) else "")
            # Always show B for block methods
            if c["sampler"] in ("bcfm_infer", "bd3lm") or model in ("M2", "M2-TS", "M3", "BD3LM"):
                title = f"{model}, B={B}, steps={steps}"
            print(f">>> {dataset} {title}", flush=True)
            t0 = time.perf_counter()
            try:
                if c["sampler"] in ("full_cfm", "bcfm_infer") and dataset == "text8" and model in ("M1", "M2"):
                    key = "text8_m1"
                    if key not in cfm_cache:
                        cfm_cache[key] = load_module(_text8_cfm_cfg(TEXT8_M1), DEVICE).eval()
                    toks = _gen_cfm(cfm_cache[key], c["sampler"], steps, B, batch=min(32, N_SAMPLES))
                    texts = _decode_text8(toks)
                elif c["sampler"] == "bcfm_infer" and model == "M3":
                    key = "text8_m3"
                    if key not in cfm_cache:
                        cfm_cache[key] = load_module(_m3_cfg(TEXT8_M3), DEVICE).eval()
                    toks = _gen_cfm(cfm_cache[key], "bcfm_infer", steps, B, batch=min(32, N_SAMPLES))
                    texts = _decode_text8(toks)
                elif c["sampler"] in ("full_cfm", "bcfm_infer") and dataset == "tinystories":
                    key = "ts_m1"
                    if key not in cfm_cache:
                        cfm_cache[key] = load_module(_ts_cfm_cfg(TS_M1), DEVICE).eval()
                    toks = _gen_cfm(cfm_cache[key], c["sampler"], steps, B, batch=min(4, N_SAMPLES))
                    texts = _decode_gpt2(toks)
                elif c["sampler"] in ("ar", "mdlm", "bd3lm"):
                    ckpt = Path(c["ckpt"]) if c.get("ckpt") else None
                    if ckpt is None or not ckpt.exists():
                        # fall back to known paths
                        if model in ("AR",):
                            ckpt = BASE / "ar_text8_seed12345/checkpoints/last.pt"
                        elif model == "AR-TS":
                            ckpt = BASE / "ar_tinystories_seed12345/checkpoints/last.pt"
                        elif model == "MDLM":
                            ckpt = BASE / "mdlm_text8_seed12345/checkpoints/last.pt"
                        elif model == "MDLM-TS":
                            ckpt = BASE / "mdlm_tinystories_seed12345/checkpoints/last.pt"
                        elif model == "BD3LM":
                            ckpt = BASE / "bd3lm_b16_text8_seed12345/checkpoints/last.pt"
                        else:
                            raise FileNotFoundError(model)
                    kwargs = {}
                    if c["sampler"] == "mdlm":
                        kwargs["num_steps"] = steps
                    if c["sampler"] == "bd3lm":
                        kwargs["steps_per_block"] = steps
                        kwargs["first_hitting"] = c["first_hitting"]
                    texts = _gen_baseline(ckpt, **kwargs)
                else:
                    raise ValueError(f"unhandled config {c}")
            except Exception as e:
                print(f"FAILED {title}: {e}", flush=True)
                texts = [f"<ERROR: {e}>"]

            dt = time.perf_counter() - t0
            sections.append("")
            sections.append("-" * 78)
            sections.append(f"## {title}")
            sections.append(f"# dataset={dataset}  sampler={c['sampler']}  exp_ref={c['exp']}")
            if c.get("gen_ppl") is not None:
                sections.append(f"# gen_ppl(seed0 eval)={c['gen_ppl']:.2f}")
            sections.append(f"# fresh_seed={SEED}  n={len(texts)}  gen_seconds={dt:.2f}")
            for i, text in enumerate(texts, 1):
                sections.append("")
                sections.append(f"[{i}]")
                sections.append(text.replace("\n", "\\n"))

            # free baseline models between calls (they are large)
            if c["sampler"] in ("ar", "mdlm", "bd3lm"):
                torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sections) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
