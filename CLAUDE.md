# BCFM — project guide (read me first)

**Block Categorical Flow Maps**: semi-autoregressive, few-step text generation. We fork
the Categorical Flow Maps codebase (`semicat`) and add block-by-block generation. Full
research plan + methodology: `paper/main.tex`. Curator: Sergei Kholkin (Skoltech).

## Repository origin
This repo is a fork of **`github.com/olsdavis/semicat` @ `558602a`** (an ashleve
`lightning-hydra-template`: PyTorch-Lightning + Hydra). The base is the first commit; the
`upstream` git remote points at it. **All our code lives in `block/` and `eval/`** — the
vendored `semicat/` package stays as close to upstream as possible so diffs and merges stay
clean.

```
block/      our core: nfe, blockwise sampling, checkpoint-compatible M3 net + module
eval/       standalone eval harness: generate -> metrics -> results/<exp>/metrics.json
semicat/    vendored upstream (touch minimally, guard changes behind flags)
configs/    Hydra configs (see below)
results/    committed metrics.json per run (small; paper tables/plots read these)
paper/      the .tex sources
```

## The three models (see paper/main.tex §Experiments)
- **M1** full-sequence CFM (baseline) — `experiment=cfm_text8_baseline`.
- **M2** inference-time BCFM — training-free; wrap an M1 checkpoint (`sampler=bcfm_infer`).
- **M3** training-time BCFM — block-causal model initialized strictly from the trained M1,
  `experiment=bcfm_finetune_text8`, then evaluated with
  `sampler=bcfm_train model=block_text8`.

## Environment
GPU/cluster boxes:
```sh
mamba env create -f environment.yaml && mamba activate semicat
```
`.env` (never committed) holds `PROJECT_ROOT`, `COMET_API_KEY`, `COMET_WORKSPACE`, and
`DATASET_CACHE_DIR`. Text8 data goes in `data/text8/{train,val}.bin, meta.pkl` (download per
upstream README). **Mac/CPU cannot even import the text8 net** (`semicat/net/duo.py` imports
`flash_attn` at module level, linux+cu124 wheel only): locally you can run `pytest tests/`
(pure `block/` + torch) and Hydra config-compose checks, nothing that touches `semicat`.
All real runs happen on GPU boxes.

## Running things
```sh
# Train M1 baseline (quality is measured separately, by the eval harness)
python -m semicat.train experiment=cfm_text8_baseline trainer=gpu logger=comet

# Evaluate: one sampler setting -> one results/<exp>/metrics.json
python -m eval.run_eval ckpt_path=/path/to/m1.ckpt sampler=full_cfm sampler.steps=4
python -m eval.run_eval ckpt_path=/path/to/m1.ckpt sampler=bcfm_infer \
    sampler.block_size=16 sampler.steps_per_block=2

# Sweep (multirun): each job writes its own results file — no collisions
python -m eval.run_eval -m ckpt_path=/path/to/m1.ckpt sampler=bcfm_infer \
    sampler.block_size=4,8,16,32 sampler.steps_per_block=1,2,4

# Fine-tune M3 from baseline/s_baseline.ckpt and evaluate it
./scripts/train_bcfm_from_baseline.sh
python -m eval.run_eval ckpt_path=/path/to/m3.ckpt sampler=bcfm_train model=block_text8 \
    model_id=M3 sampler.block_size=16 sampler.steps_per_block=2

# Cloud / SLURM: add hydra/launcher=slurm to a multirun (needs hydra-submitit-launcher)
```
Local = Hydra's default launcher (nothing to configure). Trackers: `logger=comet`.

## Team workflow (4–5 people, minimize merge pain)
1. **Branch per person/feature**, PR into `main`. `block/` and `eval/` are **shared infra** —
   change them only via reviewed PR, keep them small and stable.
2. **One YAML per experiment, never edit someone else's.** Name experiment configs
   `configs/experiment/<model>_<tag>__<owner>.yaml` (e.g. `bcfm_bsweep__misha.yaml`). New
   samplers go in `configs/sampler/`. Append-only ⇒ no line-level conflicts.
3. **Never commit runs.** `outputs/`, `logs/`, `*.ckpt`, `data/` are gitignored; they live in
   Comet + local artifact dirs. Every run auto-saves its resolved config under `logs/`.
4. **Commit results as small JSON** (schema below). Distinct paths ⇒ trivial merges.
5. **Register your runs in `EXPERIMENTS.md`** (append a row: name, owner, status, Comet link).

## results/<exp>/metrics.json schema (frozen — don't break downstream plots)
```json
{ "model": "M2", "sampler": "bcfm_infer", "block_size": 16, "steps_per_block": 2,
  "nfe_per_token": 0.125, "flop_cost_per_token": 32.0, "cost_model": "full_recompute",
  "seed": 0, "prefix": "generated|gold", "prefix_mode": "clean|renoise",
  "discretize": "argmax|sample", "schedule": null, "report_block_size": 16,
  "entropy_per_block": [/* pooled, nats */], "entropy_per_block_ps": [/* per-sample */],
  "gen_ppl": 0.0, "sampling_seconds": 0.0, "tokens_per_sec": 0.0,
  "ckpt": "...", "commit": "<sha>", "n_samples": 256 }
```
`block/nfe.py` is the single source of truth for `nfe_per_token` / FLOP cost (read its
docstring: M2 forwards cost as much as M1's, M3's are cheaper). Full-sequence CFM is the
`block_size == length` case (one block). `sampler=gold` scores real data — the reference
lines for every figure.

## Eval gotchas
- **Checkpoint loading is strict**: `eval_bcfm.yaml`'s `model.net` block must match the
  architecture the checkpoint was trained with (`length`, `embed_type`, ...). Defaults
  mirror `cfm_text8_baseline`; if you trained something else, override `model.net.*` to
  the values in that run's resolved config (in its `logs/` dir).
- **Every swept axis must show up in `exp_name`**, or two multirun jobs overwrite each
  other's results dir. The default template covers B/steps/prefix/prefix_mode/discretize/
  seed; custom `sampler.schedule` runs must set `exp_name` manually.
- With `sampler.prefix=gold`, returned sequences are still the *generated* blocks (each
  conditioned on the gold prefix) — that's the exposure-bias probe, not a copy of the data.

## M3 internals (implemented — `block/`)
- `block/mask.py` — BD3-LM block-causal 2L×2L mask (the correctness-critical piece: a noisy
  block sees only *strictly previous* clean blocks, never its own clean copy), exposed both
  as a dense reference and the official-style compiled FlexAttention mask.
- `block/block_dit.py` — `BlockDIT`: DiT over `[x_clean; z_noisy]` with the mask + per-token
  (s_b, t_b) adaLN and the exact trainable namespace of upstream `duo.DIT`. Ordinary
  VFM/teacher passes use compiled BD3-LM-style FlexAttention; CFM/ECLD JVP passes cache the
  clean stream and run SemiCat's custom Triton JVP kernel on each permitted sparse context.
- `block/block_semicat.py` — block-factorized VFM plus either the checkpoint's CFM/
  Lagrangian objective (`sd_type=lag`, default) or untouched-upstream ECLD.
- `block/checkpoint.py` — strict network-only initialization from M1, with source
  loss/shape validation and a JSON provenance report. Lightning resume remains a separate
  full-state path.
- `block/sampling.py::block_causal_sample` — M3 sampler (block by block through the masked
  forward). CPU fallbacks are tested, and CUDA tests cover Flex/JVP/backward and `B=L`
  equivalence to untouched CFM.
