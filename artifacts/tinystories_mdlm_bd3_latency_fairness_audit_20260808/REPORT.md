# MDLM vs BD3-LM latency fairness audit

Date: 2026-08-08

## Question and fairness criterion

The audit asks whether the plotted TinyStories MDLM and BD3-LM sequence
latencies measure comparable end-to-end inference work, and whether a
one-sided optimization changes the ranking. A protocol is considered matched
when hardware, precision, batch/length, compile policy, timing boundaries,
sampling policy, checkpoint scale, and quality pairing agree. A
"symmetrically maximum-speed" claim additionally requires every exact shared
optimization to be applied to both implementations.

## Validity checks

- Both stored curves use one NVIDIA H100 NVL, BF16 parameters/computation,
  batch 1, length 256, `torch.compile(mode="max-autotune")`, five warmups, and
  30 repeats.
- Both runners execute the same AST-loaded upstream timer (SHA-256
  `120e2e53826dc540d1c7372049c1582c8bd9af679c4412489021b0ebdd3bce71`),
  which synchronizes CUDA immediately before and after every timed generation.
- Both checkpoints are step 100k EMA checkpoints with 51,119,442 trainable
  parameters, the same 3,276,800,000 processed training tokens, and
  `time_conditioning=false`.
- Both timed paths include top-p 0.9 sampling, RNG, mask updates, and output
  construction. Metric-only entropy/mask-count diagnostics are omitted from
  both.
- The BF16 quality points were regenerated with the corresponding fast paths;
  FP32 quality was not attached to BF16 latency.

Primary stored evidence:

- `../tinystories_m1_mdlm_h100_bf16_maxautotune_fair_20260808/m1_mdlm_latency.json`
- `../tinystories_fused_sampler_h100_bf16_maxautotune_fair_20260807/fused_sampler_latency.json`
- `../../dumps/ts_MDLM_fused_fast_bf16_maxautotune.json`

## Competing fairness mechanisms and predictions

1. **Matched end-to-end implementations.** Prediction: the common protocol and
   sampler settings agree, and at equal approximate call counts the
   current-block BD3 backend remains faster than the full-sequence MDLM
   backend.
2. **MDLM unchanged-state cache advantage.** Prediction: high-N MDLM points
   execute fewer than N+1 denoisers and disabling the cache raises latency;
   N=16/32 should be unchanged if they have no cache hits.
3. **BD3-specific KV/fusion "cheat".** Prediction: the optimization would
   alter tokens or omit required sampler work. In contrast, exact finalized
   prefix KV reuse and fusing the cache commit with the next useful score are
   legitimate if sampler outputs are preserved.
4. **Seed-selection bias.** Prediction: repeating seed 0 thirty times measures
   hardware jitter for one stochastic trajectory, while actual MDLM call count
   varies across seeds.
5. **Run/environment confounding.** Prediction: a fresh isolated diagnostic
   can differ by several percent because the two publication curves were not
   measured in one interleaved process and do not record GPU UUID/clocks/power.

## Targeted isolated diagnostic

Script: `../../eval/audit_mdlm_bd3_latency_fairness.py`

Artifact: `audit.json`

The final artifact is an isolated H100 BF16/max-autotune run over seeds 0--4,
with three synchronized repeats per seed. An earlier contended run was
overwritten and is not evidence.

### MDLM cache ablation

| N | cached median (ms) | no-cache median (ms) | cached calls over seeds | latency reduction |
|---:|---:|---:|---|---:|
| 16 | 31.334 | 31.366 | 17,17,17,17,17 | 0.1% |
| 32 | 61.621 | 61.577 | 33,33,33,33,33 | none |
| 64 | 121.038 | 121.353 | 64,65,65,64,65 | 0.3% |
| 128 | 232.761 | 241.966 | 108,112,108,113,109 | 3.8% |
| 256 | 440.405 | 481.113 | 168,164,169,167,169 | 8.5% |

The cache is therefore material only for the high-N MDLM tail. It does not
explain MDLM N=16/32, including the paper's principal low-latency MDLM point.
The cache is exact because the model has no time conditioning and reuses logits
only when the token state did not change. However, ancestral BD3 has the same
exact reuse opportunity on a no-reveal step and currently does not exploit it.
First-hitting BD3 always reveals one token, so this particular cache does not
apply to its 256-call point.

### BD3 diagnostic review

The isolated multi-seed medians were 19.77, 36.59, 66.41, and 111.63 ms for
S=1,2,4,8. S=1/S=8 agree with the stored 30-repeat curve within 1%, while S=2
and S=4 are about 9--10% slower than the stored values. This does not support
replacing the publication curve; it is evidence that a definitive few-percent
claim needs one interleaved runner with full environment capture.

## Excluded explanations

- No asynchronous CUDA work escapes the timed interval.
- Sampling and mask transitions are not excluded from either timed path.
- Model size, checkpoint step, EMA policy, precision, batch size, and sequence
  length do not explain the MDLM/BD3 difference.
- BD3's finalized-prefix KV cache is not itself a cheat: the prefix cannot
  change, and MDLM's globally bidirectional changing state does not admit the
  same prefix cache.
- At matched approximate call counts, BD3 is faster in the stored curve:
  19.61 vs 32.18 ms near 16 calls, 33.59 vs 62.30 ms near 32 calls, and
  60.14 vs 121.55 ms at 64 calls. MDLM's end-to-end frontier advantage comes
  from reaching useful quality in fewer global steps, not from being faster per
  call.

## Remaining gaps

- The stored BD3 parity artifact tests an older fast-source hash (`4191...`),
  while the measured BF16 path uses `f5c356...`; no stored current-source MDLM
  parity artifact exists.
- Both publication latency runners repeat seed 0 rather than sampling a
  distribution of sequences. MDLM N=128 seed 0 is among the lower-call
  trajectories.
- Ancestral BD3 lacks the exact unchanged-state cache used by MDLM.
- The two curves were run on different days/jobs and do not record GPU UUID,
  clocks, power state, driver, or complete software environment.
- A "backbone call" is not equal work: MDLM scores 256 positions, whereas BD3
  scores a 16-token active block (32 tokens in the fused transition).
- The fast runners are presently untracked files; stored source hashes identify
  them, but the recorded repository commit alone cannot reproduce them.

## Supported conclusion

The basic H100 end-to-end protocol is comparable and there is no evidence of a
hidden optimization that makes BD3 unfairly fast. The stronger claim that both
models are symmetrically maximum-speed is not established. The confirmed
one-sided optimization is MDLM's unchanged-state cache, which improves N=128
by about 3.8% and N=256 by about 8.5% in the isolated ablation, while leaving
N=16/32 unchanged. A definitive publication comparison should add the same
exact cache to ancestral BD3 (or report MDLM cache-off), persist current-source
parity, and rerun both models interleaved over multiple seeds.
