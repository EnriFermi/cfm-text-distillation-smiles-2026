# eval test — MDLM vs CFM vs BD3-LM generation latency

Self-contained latency benchmark for the three TinyStories checkpoints in
`../2 models/`. Nothing here writes to either model repository; they are only
put on `sys.path`.

## What is being compared

|                | CFM (`cfm_mod/last.ckpt`) | BD3-LM (`bd3lm/last.pt`) | MDLM (`mdlm_.../checkpoints/last.pt`) |
|----------------|---------------------------|---------------------------|---------------------------------------|
| repo           | `../cfm-text-distillation-smiles-2026` | `../BCFM Baselines` | `../BCFM Baselines` |
| kind           | M3 block-causal | block diffusion | full-sequence masked diffusion |
| backbone       | 384 / 6 / 6 | 384 / 6 / 6 | 384 / 6 / 6 |
| params         | 51.96 M | 51.12 M | 51.12 M |
| length / block | 256 / 16 | 256 / 16 | 256 / full sequence |
| tokenizer      | GPT-2 BPE (50257) | GPT-2 BPE (50257) | GPT-2 BPE (50257) |
| train steps    | 100 000 | 100 000 | 100 000 |

Same data, tokenizer, sequence length, and backbone width/depth. For matched
scheduled denoiser calls, `S` steps per each of 16 CFM/BD3 blocks is compared
with `16*S` full-sequence MDLM steps.

## Files

- `common.py` — `sys.path` bootstrap, the sync-bracketed `timed()` helper, and
  the row/JSON schema every script writes.
- `shim/flash_attn/` — pure-torch stand-in for flash-attn. `semicat/net/duo.py`
  imports flash-attn at module level but only calls
  `apply_rotary_emb_torch`, which is itself a pure-torch reference function, so
  no CUDA build is needed to run inference on sm_120.
- `inspect_ckpts.py` — dumps what each checkpoint actually contains.
- `bench_cfm.py` / `bench_bd3lm.py` / `bench_mdlm.py` — the three sweeps.
- `mdlm_fast.py` — lean MDLM generation and the maximum-speed active-position
  vocab/sampling path.
- `run_three_speed.py` — runs all three fastest profiles and builds the report.
- `analyze_three.py` — matched-NFE three-model table.
- `analyze.py` — fits BD3-LM's affine cost curve and inverts it to answer
  "which `steps_per_block` makes BD3-LM as fast as CFM".
- `analyze_fair.py` — the equal-footing view: BD3-LM re-measured with
  `use_kv_cache=False`, which is the setting that matches CFM (whose sampler has
  no cache at all). Also prices what the cache is worth on its own.
- `cfm_fast.py` — an exact, much faster CFM block-causal sampler. The reference
  one runs both vocab-sized projections over all 512 doubled-stream positions
  when only 16 are used; this one does the clean stream as an embedding lookup,
  the noisy stream on the active block only, and the vocab head on the active
  block only. Token-for-token identical to the reference in fp32
  (`assert_matches_reference`).
- `analyze_opt.py` — the third regime: every optimization on, both sides.
- `sample_texts.py` — decodes a few samples from each model, so the timings are
  demonstrably of a working generator.
- `results/` — the JSON sweeps plus `comparison.md`.

## Running

```sh
python run_three_speed.py

# Individual reference/eager sweeps:
python bench_cfm.py   --nfe 1 2 4 8 16      --batch-size 1 4 16 32
python bench_bd3lm.py --steps-per-block 1 2 4 8 16 32 --batch-size 1 4 16 32 --first-hitting
python bench_mdlm.py  --num-steps 16 32 64 128 256 --batch-size 1 4 16 32
python analyze.py

# equal-footing pass: strip BD3-LM's only inference acceleration
python bench_bd3lm.py --steps-per-block 1 2 4 8 16 32 --batch-size 1 4 16 \
    --no-kv-cache --out bd3lm_latency_nocache.json
python analyze_fair.py
```

Needs torch with a CUDA build matching the GPU (sm_120 here → cu128), plus
`lightning torchmetrics torchdiffeq omegaconf hydra-core einops jaxtyping
transformers rich wandb rootutils tiktoken numpy<2`. The conda environments on
this machine have torch but not `lightning`/`torchdiffeq`/`wandb`/`rootutils`.

## Method

Every point is the median of 5 timed runs after 2 warmups, with
`torch.cuda.synchronize()` on both sides of the timed region. The warmups matter
here: CFM's block-causal net compiles a FlexAttention mask on first use.
