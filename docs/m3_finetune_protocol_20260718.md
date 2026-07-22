# M3 fine-tune protocol — run of 2026-07-18 (pre-loader-fix)

**Status: completed, but not a valid measurement of H1.** See §10. This document
is a record of what was actually run, reconstructed from the run directory, the
resolved Hydra config, the checkpoint metadata and the analysis artifacts. It is
not a recommendation to repeat this configuration unchanged.

This run predates the `split_units` loader fix (`configs/data/text8.yaml`), so it
used the upstream character-slicing loader throughout.

---

## 0. Provenance

| | |
|---|---|
| Run directory | `logs/train/2026-07-18_10-46-57_12345_local/` |
| Started | 2026-07-18 10:46:58 |
| Finished | 2026-07-19 17:28:05 |
| Wall clock | 30 h 41 min, 100,000/100,000 steps completed |
| Comet | `kind_angler_5940` / `cef17582c0174d4a99bdb558a304f890` |
| Repo commit | `57e941b` |
| Launch overrides | `experiment=bcfm_finetune_text8`, `trainer=gpu`, `logger=comet` — nothing else |

Because there were no further CLI overrides, the effective configuration is
exactly `configs/experiment/bcfm_finetune_text8.yaml` composed with
`configs/model/block_text8.yaml`; the resolved copy is in
`logs/train/2026-07-18_10-46-57_12345_local/.hydra/config.yaml`.

`logs/train/2026-07-18_10-46-02_12345_local/` is a false start 56 seconds
earlier — its log stops at model instantiation and it has no checkpoints. It is
not part of this experiment.

Artifacts produced: 11 checkpoints (`step_0010000.ckpt` … `step_0100000.ckpt`
plus `last.ckpt`, ~1.11 GB each) and `cfm_initialization.json`.

---

## 1. Objective

Test hypothesis **H1** of `paper/main.tex`: the training-time block adaptation
(M3) beats the training-free inference-time wrapper (M2). The model is
initialized from the trained full-sequence CFM and fine-tuned into a genuinely
block-conditional categorical flow map.

## 2. Initialization

Network-only strict load from `baseline/s_baseline.ckpt` via
`block/checkpoint.py`:

- 164 tensors, 92,491,547 parameters loaded;
- source `global_step` 390,000, `epoch` 142, `in_shape` `[256, 27]`,
  `sd_type` `lag`; checkpoint sha256 `38ad1af5aba40ce1086a345d652a004ba627d5eff6e6a522fb66e0adc6ae2147`;
- the loader asserts `source_sd_type == expected_source_sd_type` (`lag`).

Optimizer, scheduler and step counter start fresh (`ckpt_path: null`). This is
initialization, **not** a Lightning resume. Provenance is written to
`cfm_initialization.json` in the run directory.

## 3. Data

Text8 character level, `data/text8/{train,val}.bin`, vocabulary 27, sequence
length `L = 256`, stride-one windows.

Configured `train_val_test_split: [351563, 20000, 19063]`. **The upstream loader
reads these as raw character offsets, not as counts of 256-token sequences**, so
the actual exposure was:

| split | configured | characters used | stride-one windows |
|---|---:|---:|---:|
| train | 351,563 | 351,563 of 90,000,000 | 351,308 |
| val | 20,000 | 20,000 | 19,745 |
| test | 19,063 | 19,063 | 18,808 |

- Train corpus coverage **0.3906%**; adjacent windows overlap by 255/256 tokens;
  roughly 1,373 disjoint-window equivalents against 92.5M parameters.
- Validation is unshuffled, 155 batches.

Recorded in `artifacts/bcfm_val_loss_diagnosis_20260719/data_geometry.json`.
This is an inherited upstream defect: both `semicat/data/text8.py` and the split
values match `olsdavis/semicat@558602a` byte-for-byte, and commit `57e941b` did
not touch the loader.

## 4. Architecture

`block.block_dit.BlockDIT` — hidden 768, 12 layers, 12 heads, `cond_dim` 128,
dropout 0.1, `embed_type: naive`, vocab 27, `block_size` 16 (so 16 blocks of 16
tokens over L = 256).

The trainable namespace is identical to upstream `duo.DIT`, which is what makes
the strict load possible. Input is the doubled stream
`[x_clean; z_noisy]` of length 512 under the `2L×2L` mask of `block/mask.py`:

```
clean -> clean   block-causal        (bk <= bq)
noisy -> clean   strictly previous   (bk <  bq)
noisy -> noisy   own block only      (bk == bq)
```

The clean half is tagged `s = t = 1` at every position; per-token `(s_b, t_b)`
reach the network only through adaLN modulation.

Attention backends: `flex` (compiled FlexAttention, kernel block size 64) for
ordinary VFM/teacher forwards; `jvp_attention_backend: auto` → SemiCat's Triton
JVP kernel for the self-distillation pass, because FlexAttention cannot be
transformed by `torch.func.jvp` in the pinned Torch. `compile: false`.

## 5. Training objective

Prior `z_0 ~ N(0, I)` on `(128, 256, 27)`, `prior_type: gaussian`.

Times are drawn **independently per block and per sample**: `t_b ~ U[0,1)`,
`s_b = t_b · U[0,1)`, with `time_eps = 0.0` (no ε cutoff, exact upstream
`torch.rand` semantics). They are broadcast to all 16 tokens of their block via
`repeat_interleave`.

The batch of 128 is split into disjoint parts —
`sd_split = int(0.25 × 128) = 32`:

- **96 samples → VFM.** Interpolant `z_t = (1-t)·z_0 + t·δ_x`; one masked forward
  with `s = t = t_b` on the noisy half and `s = t = 1` on the clean half;
  cross-entropy on the noisy half against the clean tokens;
  `label_smoothing = 0.0`.
- **32 samples → self-distillation**, `sd_type: lag` — the upstream
  CFM/Lagrangian objective, **not** ECLD:

  L_lag = ‖ (1−t)·∂_t X_{s,t} − sg q_{t,t}(X_{s,t}) + X_{s,t} ‖²

  summed over the vocabulary axis, mean-reduced. The clean-stream KV is computed
  once outside `torch.func.jvp` with its reverse-mode graph retained; the JVP is
  taken with respect to `t`.

Total loss is the unweighted sum `total = vf_loss + sd_loss`, i.e. λ = 1.

> Deviation from the paper: `paper/main.tex` Eq. (total) prescribes
> `L_VFM + λ·L_ECLD` with the 4/2 coefficients inside ECLD, evaluated on the same
> batch. This run used the Lagrangian loss on a disjoint 25% batch split with
> λ = 1. The ECLD branch exists (`block/block_semicat.py::_ecld_model_step`) but
> was not exercised — that is ablation E7 in `docs/experiment_plan.md`.

## 6. Optimization

| | |
|---|---|
| Optimizer | AdamW, `lr = 1e-4`, `weight_decay = 0.01` |
| Scheduler | **none** — constant LR for all 100k steps |
| Precision | `bf16-mixed` |
| Gradient clipping | 1.0, global norm |
| Gradient accumulation | 1 → effective batch = 128 |
| Seed | 12345, `deterministic: false` |
| Sanity val steps | 0 |

## 7. Schedule and checkpointing

At 351,308 windows and batch 128 the epoch is 2,745 optimizer steps
(`drop_last=False`; confirmed against the logged epoch indices at every
validation). Therefore:

- 100,000 steps = **36.4 passes** over the truncated training prefix;
- 12,800,000 window presentations over 351,308 unique windows;
- mean throughput 1.10 s/step.

Validation every 10,000 steps. `ModelCheckpoint` with `monitor: val/loss`,
`save_top_k: -1`, `save_last: true`, `filename: step_{step:07d}` — every periodic
snapshot retained, not only the best.

## 8. Environment

Single NVIDIA H100 NVL, torch 2.7.1+cu126, CUDA 12.6, Lightning 2.x,
`semicat` conda environment. Loader: `num_workers: 4`, `pin_memory: true`,
`prefetch_factor: 4`.

## 9. Results

Training loss fell to ≈0.113. **Validation loss rose monotonically**
(`artifacts/bcfm_val_loss_diagnosis_20260719/val_events.csv`):

| step | epoch | val total | val VFM CE | val Lag SD | preceding train loss |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 3 | 3.17120 | 3.10186 | 0.06933 | 0.22566 |
| 20,000 | 7 | 4.42996 | 4.35116 | 0.07881 | 0.15374 |
| 30,000 | 10 | 4.32864 | 4.25200 | 0.07664 | 0.13481 |
| 40,000 | 14 | 4.69660 | 4.61893 | 0.07767 | 0.12385 |
| 50,000 | 18 | 4.80325 | 4.72035 | 0.08289 | 0.11892 |
| 60,000 | 21 | 4.79194 | 4.70626 | 0.08569 | 0.11692 |
| 70,000 | 25 | 4.93683 | 4.84439 | 0.09244 | 0.11521 |
| 80,000 | 29 | 4.81155 | 4.72521 | 0.08634 | 0.11376 |
| 90,000 | 32 | 4.86064 | 4.77165 | 0.08899 | 0.11332 |

From 10k to 90k the total rose by 1.6894, of which VFM CE contributes 98.84%
(1.6698) and Lag SD 0.0197. The values exceed the uniform ceiling
`ln 27 = 3.296`.

## 10. Threats to validity

**This run does not measure H1.** The established mechanism is
prefix-conditioned memorization on a 256-times collapsed support.

*Fixed-input probe* (identical examples, cached Gaussian `x0` and block times at
every checkpoint; `train_unseen_tail` samples the 89.65M characters the run never
loaded):

| checkpoint | seen train-prefix CE | unseen train-tail CE | validation CE |
|---|---:|---:|---:|
| source weights | 1.346 | 1.715 | 1.725 |
| 10k | 0.177 | 3.024 | 3.012 |
| 20k | 0.131 | 4.182 | 4.207 |
| 90k | 0.116 | 4.634 | 4.643 |

The unseen tail and the validation split track each other almost exactly, so the
failure follows membership in the truncated training prefix, not the
train/val file boundary.

*Block localization at 90k*: block 1 (no prefix available) holds CE 1.776 on the
seen prefix; seen-prefix block 2 is 0.0795; seen-prefix blocks 3–16 are
approximately 1e-6; unseen/validation blocks 2–16 are 3.3–5.6. This is the
signature of using the clean prefix as a lookup key for a memorized
continuation. It also explains the apparent training plateau:
1.776/16 = 0.111 ≈ the measured 0.116.

*Calibration at 90k on fixed validation inputs*: accuracy 0.498 → 0.388, mean
maximum confidence 0.490 → 0.807, confidence on wrong predictions 0.239 → 0.731,
predictive entropy 1.759 → 0.577. The model sharpens around memorized but
incorrect continuations.

**The judge-metric gain is an artifact of memorization.** Generative NLL against
GPT-J-6B, 2,048 sequences per point, seed 0, argmax discretization
(`artifacts/bcfm_generation_nll_b16_initial_vs_90k_20260719/`):

| NFE | source NLL | 90k NLL | source PPL | 90k PPL | ΔNLL [95% CI] | ΔNLL excluding 90k outputs with ≥64-char exact train match |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.74689 | 6.59311 | 851.41 | 730.05 | −0.15378 [−0.18011, −0.12849] | **+0.00908** [−0.00872, +0.02679] |
| 16 | 6.75953 | 6.58215 | 862.24 | 722.09 | −0.17739 [−0.20625, −0.14900] | **+0.02792** [+0.00922, +0.04656] |
| 32 | 6.77028 | 6.57796 | 871.55 | 719.07 | −0.19232 [−0.22237, −0.16256] | **+0.02474** [+0.00505, +0.04469] |

Memorization audit by exact suffix-automaton matching against the only 351,563
exposed training characters:

| NFE | source ≥64-char match | 90k ≥64-char | 90k ≥128-char | 90k exact 256-char windows |
|---:|---:|---:|---:|---:|
| 8 | 0.0% | 13.92% | 13.43% | 1.37% |
| 16 | 0.0% | 16.65% | 16.11% | 1.81% |
| 32 | 0.0% | 17.43% | 16.70% | 1.90% |

Excluding the matched outputs flips the sign: the aggregate 14–17% PPL
improvement does not survive.

**Confounds not separated.** Constant `lr = 1e-4`, a fresh AdamW and no scheduler
plausibly accelerate specialization; relative parameter drift from the source is
19.6% at 10k, 31.3% at 30k and 47.5% at 90k. Their independent contribution
needs a matched LR/optimizer ablation on corrected data. Likewise, a small Lag-SD
scalar does not prove a small Lag-SD gradient — that needs a same-start
`sd_prop=0` versus `lag` ablation.

**The source model shares the defect.** `baseline/s_baseline.ckpt` was itself
trained with the same truncated loader, so a clean comparison requires retraining
the source CFM too, not just re-running the fine-tune.

## 11. Reproduction

As run (upstream truncating loader, since `split_units` did not exist yet):

```sh
./scripts/train_bcfm_from_baseline.sh
# = python -m semicat.train experiment=bcfm_finetune_text8 trainer=gpu logger=comet
```

To repeat the same protocol on the full corpus, add the loader fix:

```sh
./scripts/train_bcfm_from_baseline.sh data.split_units=sequences
```

Note that `data.split_units=chars` (the default) reproduces the run documented
here. Re-running with `sequences` changes the training distribution by a factor
of 256 in support size and is a different experiment, not a bug-fix rerun of this
one.

## 12. Related artifacts

- `artifacts/bcfm_val_loss_diagnosis_20260719/` — why validation loss rose;
  `val_events.csv`, `fixed_probe*.csv`, `data_geometry.json`, `REPORT.md`.
- `artifacts/bcfm_generation_nll_b16_initial_vs_90k_20260719/` — source weights
  versus 90k on generative NLL, with the memorization audit.
- `results/bcfm_step30000_cfm_vs_blockwise_gptj6b_n2048_seed0/` — full-CFM versus
  blockwise at the 30k checkpoint, GPT-J-6B judge, NFE 1/2/4/8, 2,048 samples.
- `artifacts/bcfm_verification/` — pre-run correctness evidence (mask no-leak,
  CUDA JVP, `B = L` equivalence, strict checkpoint load).

## 13. Documentation drift

`docs/bcfm_finetune.md` still states that "No optimization experiment was
launched and no optimizer step was performed". That was written before this run
and is stale.
