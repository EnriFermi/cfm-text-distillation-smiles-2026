# BCFM experiment plan — enough results for the paper

Goal: every claim in `paper/main.tex` is backed by a figure or table with **error bars**
and a **stated decision rule**, plus the two headline curves and ablations that isolate each
design choice. This doc is the source of truth for *what to run*; `EXPERIMENTS.md` tracks
*who ran what*; `results/<exp>/metrics.json` holds the numbers; `eval/aggregate.py` turns
those into the paper's tables/figures.

---

## 1. Claims inventory (extracted from main.tex)

| # | Claim | Where | Type |
|---|-------|-------|------|
| C1 | Blockwise BCFM has a **better NFE/quality tradeoff** than full-sequence CFM | Abstract, §Exp **H2** | headline hypothesis |
| C2 | **Training-time** BCFM beats the **training-free** adaptation | §Exp **H1** | headline hypothesis |
| C3 | Block size has a **sweet spot** (B∈{4,8,16}); limits: **B=1**≈AR-latent, **B=L**=full CFM | §Methods | design claim + sanity |
| C4 | The central knob is **B × steps-per-block** | §Methods, §Sampling | design claim |
| C5 | **Entropy collapse compounds across blocks** (risk to characterize) | §Sampling | risk / diagnosis |
| C6 | **Exposure bias**: at inference the prefix is *generated*, not gold (risk) | §Sampling | risk / diagnosis |
| C7 | Finalized blocks form a **reusable KV cache** ⇒ fast, **variable-length** generation | §Bg, §Sampling | capability claim |
| C8 | ECLD self-distillation is what enables **few-step** (1–4) block generation | §Bg, §Methods | mechanism claim |

Success = each of C1–C8 has an artifact + decision rule below. C1/C2 are the paper's spine.

---

## 2. Experiment catalog

Models: **M1** full-seq CFM · **M2** inference-time KV-cached BCFM (wrap M1) ·
**M3** training-time BCFM.
Default data: text8, L=256. Judge: gpt2-large gen-PPL (primary), GPT-J-6B (one confirm table).
Samples: **256** for sweeps, **512** for final headline points. Seeds: see §4.

| ID | Claim(s) | Model(s) | Sweep axes | Primary metric | Decision rule / what it shows | Artifact |
|----|----------|----------|-----------|----------------|-------------------------------|----------|
| **E0** | prereq | M1 | steps {1,2,4,8,16} | gen-PPL | sanity: PPL improves with steps (matches CFM) | Tab. baselines |
| **E0-ref** | all figs | – (data) | `sampler=gold` | gen-PPL + entropy of real text8 | reference lines: the PPL floor and entropy ceiling every figure is read against | dashed line in Figs. 1–5 |
| **E1** | C1 (H2) | M1,M2,M3 | full NFE grid (see §3) | gen-PPL vs **network forwards/token** | H2 holds if M2/M3 Pareto frontier lies below M1's over some NFE range **and** those points keep per-sample entropy within ~10% of the E0-ref value (a PPL win from entropy collapse is not a win); report crossover | **Fig. 1** (headline) |
| **E2** | C2 (H1) | M2,M3 | matched (B,steps/blk) | Δ gen-PPL (paired) | H1 holds if M3 < M2 at equal NFE on majority of grid; paired test | **Tab. H1** |
| **E2b** | C2 control | M1+ (train) | continue M1 for the same extra steps as the M3 finetune | gen-PPL | guards H1 against "M3 is better only because it trained longer": M3 must also beat the compute-matched M1+ | row in Tab. H1 |
| **E3** | C3 | M2,M3 | B {1,2,4,8,16,32,64,128,256}, steps/blk=1,2 | gen-PPL vs **B** | sweet spot = argmin region; **B=256 must ≈ M1** (sanity), B=1 = AR-latent | **Fig. 2** |
| **E4** | C4 | M2,M3 | B=16, steps/blk {1,2,4,8} | gen-PPL vs **steps/blk** | diminishing returns; where few-step saturates | **Fig. 3** |
| **E5** | C5 | M2,M3 | B{4,16}, steps/blk{1,2} | **entropy per block index** | collapse = downward trend; compare M2 (worse?) vs M3 | **Fig. 4** |
| **E6** | C6 | M2,M3 | prefix {generated, gold} | Δ gen-PPL (bias gap) | gap>0 = exposure bias; expect M2 gap > M3 gap. Note: gold-prefix output is a patchwork of independently-conditioned blocks, so its judge-PPL carries a block-boundary confound — read the *gap* and the per-block entropy curves, not the absolute number | **Tab. exposure** |
| **E7** | C8 | M3 (train) | sd_type {ecld, lag, none} | gen-PPL @1 step/blk | ECLD > VFM-only at few steps ⇒ C8 | **Tab. ablation-sd** |
| **E8** | C8 | M3 (train) | sd_prop {0, .25, .5}, λ | gen-PPL | sensitivity of self-distillation strength | Tab. ablation-λ |
| **E9** | robustness | M1/M3 (train) | prior {gaussian, discunif} | gen-PPL | prior choice effect | Tab. ablation-prior |
| **E10** | C4 / M2 heuristic | M2 | schedule {uniform, front-loaded, single-jump} @steps/blk=2 | gen-PPL | "how much is free": best (s,t) heuristic for un-trained-blockwise model | Tab. ablation-sched |
| **E11** | C5 mitig. | M2,M3 | discretize {argmax, sample} | gen-PPL + entropy | sampling vs argmax at block boundary (collapse mitigation) | Tab. ablation-disc |
| **E12** | C7 | M3 | length {256, 512, 1024} | gen-PPL + samples | M3 generates L>train-length; M1 cannot ⇒ length-flexibility | **Tab.+qual.** |
| **E13** | C7 speed | M1,M2,M3 | matched-quality points | **tokens/sec**, FLOPs | back "fast/cacheable": wall-clock, not just NFE (see §5). Timing is auto-recorded in every eval run (`tokens_per_sec` in metrics.json) | **Fig. 5** |

Qualitative samples at 1/2/4 steps/block for M1/M2/M3 → appendix table (C1/C2 color).

---

## 3. Concrete sweeps (copy-paste)

Set `CKPT=/path/to/m1.ckpt` (and later `CKPT_M3_B16=...`). All eval writes to `results/`.

```sh
# E0-ref: data reference lines (no model needed; run once)
python -m eval.run_eval sampler=gold n_samples=512

# E0 + E1(M1 arm): full-seq CFM across steps
python -m eval.run_eval -m ckpt_path=$CKPT model_id=M1 sampler=full_cfm \
    sampler.steps=1,2,4,8,16 seed=0,1,2 n_samples=512 report_block_size=16

# E1(M2 arm) + E3 + E4: inference-time BCFM grid (training-free, cheap — go wide)
python -m eval.run_eval -m ckpt_path=$CKPT model_id=M2 sampler=bcfm_infer \
    sampler.block_size=1,2,4,8,16,32,64,128,256 \
    sampler.steps_per_block=1,2,4 seed=0,1,2

# E5: entropy-per-block (already in every metrics.json; just aggregate these runs)
# E6: exposure bias
python -m eval.run_eval -m ckpt_path=$CKPT model_id=M2 sampler=bcfm_infer \
    sampler.block_size=16 sampler.steps_per_block=1,2 sampler.prefix=generated,gold seed=0,1,2

# E10: schedule heuristic for M2 (custom schedules need a manual exp_name)
python -m eval.run_eval ckpt_path=$CKPT model_id=M2 sampler=bcfm_infer \
    sampler.block_size=16 sampler.schedule="[[0,0.8],[0.8,1]]" exp_name=e10_frontload_seed0
# E11: discretization
python -m eval.run_eval -m ckpt_path=$CKPT model_id=M2 sampler=bcfm_infer \
    sampler.block_size=16 sampler.steps_per_block=1 sampler.discretize=argmax,sample seed=0,1,2

# --- M3 (needs training runs first; one per block size) ---
# E7/E8/E9 ablations are training-time:
python -m semicat.train -m experiment=bcfm_train_text8 model.sd_type=ecld,lag,none        # E7
python -m semicat.train -m experiment=bcfm_train_text8 model.sd_prop=0,0.25,0.5           # E8
python -m semicat.train -m experiment=bcfm_train_text8 model.prior_type=gaussian,discunif # E9
# E1(M3 arm)/E2/E3/E4/E12 eval each trained M3 like M2 but sampler=bcfm_train, ckpt=<M3>
```

Aggregate into paper artifacts:
```sh
python -m eval.aggregate            # -> results/summary.csv + figures/*.png
```

---

## 4. Seeds, sample sizes, and stats (so error bars are honest)
- **Sampling seeds**: 3 (0,1,2) for all headline points → error bars from sampling stochasticity (cheap, one trained model).
- **Training seeds**: 2 for the single headline config only (M1, and M3 @B=16) → reports train-seed variance; other trained models single-seed (budget). State this as a limitation.
- **Sample count**: 512 for Fig. 1/Tab. H1 final numbers; 256 for wide sweeps.
- **Paired comparison (E2)**: same seeds + same NFE grid; report mean Δ and sign-consistency across grid points (not a heavy significance test — "all relative across our own runs" per the proposal).

## 5. NFE and compute accounting
`block/nfe.py` distinguishes flow-map evaluations from all transformer invocations:
- `flow_nfe_total` counts only denoising calls.
- `cache_encode_forwards` counts the non-final clean blocks encoded to construct a
  KV prefix.
- `nfe_total` is their sum and is the Fig. 1 axis. It is the only NFE reported as a
  total number of forwards.

M2 and cached M3 process only the current block while reading clean K/V context. Fig. 1b
therefore reports `context_token_cost_per_token`, an analytical context-token proxy that
also includes cache construction; it is not measured FLOPs. Fig. 5 reports measured
tokens/sec and is the hardware-level comparison. If conclusions differ by axis, report it
explicitly.

The harness measures tokens/sec honestly: a throwaway warmup batch absorbs cudnn autotune +
torch.compile, and the timed region is bracketed by device synchronize; the RNG is reset
after warmup so the scored samples are identical with or without it. For clean numbers still
run E13 alone (fixed batch_size, an idle GPU), not inside a big multirun.

## 6. Compute budget & prioritization
- **Must-have (paper stands on these)**: E0, E0-ref, E1, E2, E2b, E3, E5, E13. M2 arm is cheap
  (no training); M3 needs ~3 training runs (B∈{4,8,16}); E2b is one cheap continued-training run.
- **Should-have**: E4, E6, E10b, E12.
- **Nice-to-have (ablations)**: E7, E8, E9, E10, E11 — each adds a training run (E7–E9) or is
  free eval (E10, E11).
- Training-run count: M1 ×(1–2 seeds) + M1+ (E2b); M3 ×3 block sizes + E7(2) + E8(2) + E9(1)
  ≈ **9–11 trains**. All eval sweeps are sampling+scoring only.

## 7. Claim → figure/table → paper section
- C1 → Fig. 1 (+Fig. 5) → §Experiments/H2.
- C2 → Tab. H1 → §Experiments/H1.
- C3 → Fig. 2 (B=256≈M1 sanity in caption) → §Methods/limits + §Experiments.
- C4 → Fig. 3 → §Experiments.
- C5 → Fig. 4 → §Experiments (entropy collapse).
- C6 → Tab. exposure → §Experiments (limitations).
- C7 → Tab.+qual. (E12) + Fig. 5 (E13) → §Experiments (length flexibility, speed).
- C8 → Tab. ablation-sd (E7) → §Methods/§Experiments.
