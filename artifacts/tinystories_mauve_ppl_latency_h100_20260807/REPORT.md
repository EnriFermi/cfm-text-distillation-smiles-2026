# TinyStories baseline latency on matched H100 hardware

Date: 2026-08-07.

## Outcome

The uploaded step-100,000 EMA checkpoints for MDLM and BD3-LM were measured
on the same NVIDIA H100 NVL used for the CFM-family curves.  Every point uses
batch size 1, sequence length 256, FP32, five warm-up generations, and the
median of 30 repeats.  Synchronization and timing are loaded directly by AST
from upstream commit `2e21c57704ab94e131e3350ffd8286632f61a7b8`.

MDLM latency is 127.97, 248.96, 482.93, 862.34, and 1503.75 ms at NFE
16/32/64/128/256.  BD3-LM ancestral latency is 178.64, 265.02, 448.57,
835.23, 1508.32, and 3032.76 ms at NFE 1/2/4/8/16/32.  The separately
reported BD3-LM first-hitting NFE-256 point is 1551.90 ms and is not mixed
into the ancestral quality curve.

## Validity checks

- GPU recorded by the runner: `NVIDIA H100 NVL`; it was idle before launch.
- Both checkpoints report `step=100000`, TinyStories, GPT-2 vocabulary,
  hidden width 384, six layers, and six heads.
- EMA weights were loaded, not raw training weights.
- MDLM SHA-256:
  `7b5e5cfc69094e7cd17f45aa8b7f2ae911f42bd55563d361814e6efde319ba7d`.
- BD3-LM SHA-256:
  `c6d4ac8e5b265e565bed7819463a57be3058742eadb2696413cf288c0cd4644f`.
- All 12 timing rows have 30 repeats, finite values, and
  `p10 <= median <= p90`.  The largest upper-tail spread is BD3-LM NFE 16,
  whose p90 is 7.55% above its median; no point shows an invalid or extreme
  timing distribution.
- The 35-point joined plot table has one H100 hardware label, 512 quality
  samples per point, and no missing MAUVE, gen-PPL, or latency values.

## A100-to-H100 comparison

The old A100 numbers were not simply relabeled.  MDLM is 1.53--1.61x faster
on H100 across its NFE grid.  BD3-LM ancestral is 1.30--1.38x faster, and its
first-hitting point is 1.41x faster.  The different speedup ranges are
plausible because BD3-LM performs many small sequential cached forwards and is
more launch/control-flow bound than the full-sequence MDLM loop.

## Revised quality--latency interpretation

The matched-H100 comparison changes the cross-family conclusion materially:

- MDLM NFE 16 (127.97 ms, MAUVE 0.9157, gen-PPL 26.99) strictly dominates
  M3 NFE 16 (190.09 ms, MAUVE 0.7550, gen-PPL 58.13).  MDLM is 1.49x faster
  while improving both quality metrics.
- MDLM NFE 16 also strictly dominates M1-200k NFE 32 (133.79 ms, MAUVE
  0.6577, gen-PPL 67.49), at 4.5% lower latency.
- BD3-LM NFE 16 has the highest measured MAUVE, 0.9507, but at 1508.32 ms.
  Relative to MDLM NFE 64 it gains only 0.0061 MAUVE, is 3.12x slower, and
  has worse gen-PPL (21.51 versus 17.14).
- MDLM supplies the global gen-PPL/latency frontier from NFE 16 onward.
  M3 still extends the CFM-only quality frontier, but it is not on the global
  MAUVE/latency frontier once the discrete baselines are timed fairly.

## Artifacts

- `tinystories_quality_latency_by_nfe.csv`: 35 quality/latency points on H100.
- `a100_vs_h100_baseline_latency.csv`: pointwise old-A100 versus new-H100
  timings and speedups.
- `quality_vs_latency_all_models_argmax.png`: reviewed combined dashboard.
- `mauve_vs_latency_by_nfe_h100_all_models.png` and
  `genppl_vs_latency_by_nfe_h100_all_models.png`: same-H100 overlays.
- `mauve_vs_latency_by_nfe_h100_by_family.png` and
  `genppl_vs_latency_by_nfe_h100_by_family.png`: family-separated views.
- `metadata.json` and `plot.log`: selection/protocol metadata and plot log.
- Baseline timing source artifact:
  `.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/`
  `tinystories_mdlm_bd3lm_latency_h100_20260807/`.
- Reproduction runner:
  `.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/eval/`
  `measure_baseline_h100_latency.py`.
