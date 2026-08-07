# Why `val/loss_epoch` rose in the Blockwise CFM fine-tune

Date: 2026-07-19  
Run: `kind_angler_5940` / Comet experiment `cef17582c0174d4a99bdb558a304f890`  
Run directory: `logs/train/2026-07-18_10-46-57_12345_local`

## Narrow conclusion

The validation loss rose because the Blockwise CFM fine-tune memorized
clean-prefix-to-continuation mappings from a 256-times truncated, heavily
overlapping training corpus. As optimization continued, the model became
nearly deterministic on continuations belonging to the seen prefix, but
increasingly confident and wrong on unseen prefixes.

The data-loader collapse is the initiating operational defect. The internal
mechanism is prefix-conditioned memorization followed by logit sharpening.
Constant LR and a fresh optimizer can accelerate this specialization, but do
not by themselves explain the split-specific failure.

## Failure definition

The logged training objective fell to about `0.113`, while the validation
objective rose:

| Optimizer step | Validation total | VFM CE | Lag SD |
|---:|---:|---:|---:|
| 10k | 3.1712 | 3.1019 | 0.0693 |
| 20k | 4.4300 | 4.3512 | 0.0788 |
| 30k | 4.3286 | 4.2520 | 0.0766 |
| 40k | 4.6966 | 4.6189 | 0.0777 |
| 50k | 4.8032 | 4.7204 | 0.0829 |
| 60k | 4.7919 | 4.7063 | 0.0857 |
| 70k | 4.9368 | 4.8444 | 0.0924 |
| 80k | 4.8116 | 4.7252 | 0.0863 |
| 90k | 4.8606 | 4.7716 | 0.0890 |

From 10k to 90k, the total rose by `1.6894`. VFM CE contributed `1.6698`,
or `98.84%`, while Lag SD contributed `0.0197`.

Evidence:

- [`val_events.csv`](./val_events.csv)
- [`mechanism_evidence.json`](./mechanism_evidence.json)
- [`loss_dynamics.png`](./loss_dynamics.png)

## Validity checks

### Metric aggregation

`val/loss_epoch` is a real epoch aggregate, not a cumulative Comet or
TorchMetrics artifact. At all nine validations it agrees with the arithmetic
mean of the 155 raw validation batch totals to within `2e-6`. Its value also
agrees with mean VFM plus mean Lag SD.

The final short batch is weighted like a full batch by `MeanMetric`, but a
sample-weighted correction changes any point by at most `0.00167`, far below
the observed rise.

Evidence:

- [`raw_metrics.json`](./raw_metrics.json)
- [`val_events.csv`](./val_events.csv)
- [`validation_summary.csv`](./independent_objective_audit/validation_summary.csv)

### Validation randomness

Ordinary validation draws fresh Gaussian `x0`, VFM block times, and Lag
`(s,t)` pairs. This can explain some late checkpoint-to-checkpoint zig-zag,
but not the main trend:

- all `155/155` aligned validation batch positions worsened from 10k to 90k;
- a fixed probe using identical examples, Gaussian noise, and block times for
  every checkpoint reproduced the growth.

Evidence:

- [`validation_batch_deltas_10k_90k.csv`](./independent_objective_audit/validation_batch_deltas_10k_90k.csv)
- [`fixed_probe_inputs.pt`](./fixed_probe_inputs.pt)
- [`fixed_vfm_probe.py`](./fixed_vfm_probe.py)

### Checkpoint alignment

From 20k onward, `ModelCheckpoint.current_score` is stale by one validation
because checkpoint saving and validation use the same 10k cadence. This affects
the score stored in callback metadata, but not the weights: validation performs
no optimizer update, so `step_N` weights are the weights evaluated at step N.
This bookkeeping issue does not create the curve.

## Data exposure defect

The configuration values `[351563, 20000, 19063]` denote counts of
256-token sequences in the original Text8 setup. The loader instead slices the
raw token tensors at those values and then constructs stride-one windows.

Consequently the run uses:

- `351,563 / 90,000,000 = 0.3906%` of `train.bin`;
- 351,308 length-256 training windows, adjacent windows overlapping by
  `255/256` tokens;
- only about 1,373 disjoint-window equivalents;
- 19,745 validation windows drawn from only the first 20,000 validation
  characters;
- approximately 32.8 passes over the truncated train prefix by step 90k;
- about 8,389 token-window presentations per unique raw training character by
  step 90k;
- 92.5M model parameters.

Evidence:

- [`data_geometry.json`](./data_geometry.json)
- [`data_exposure.json`](./independent_objective_audit/data_exposure.json)
- `semicat/data/text8.py`, lines 17–21 and 80–96
- resolved run config: `logs/train/2026-07-18_10-46-57_12345_local/.hydra/config.yaml`

## Competing causal mechanisms and discriminating predictions

| Hypothesis | Unique prediction | Result |
|---|---|---|
| Logging or reduction bug | Raw batch means do not reproduce the epoch value, or state accumulates across validations | Excluded: raw means reproduce every point |
| Validation RNG | A fixed-data/fixed-`x0`/fixed-`t` probe removes the checkpoint trend | Excluded as the main cause: fixed probe reproduces it |
| Validation-domain peculiarity | Unseen text from the same `train.bin` remains close to seen-prefix performance | Excluded as the main cause: unseen train tail matches validation |
| General numerical corruption | All blocks and noise levels degrade, or tensors contain NaN/Inf | Excluded: tensors are finite; block 1 and high-`t` inputs do not show the same failure |
| Own-clean-block leakage | Every split becomes trivial, including block 1 or unseen continuations | Excluded: the mask is strict and the observed block signature is the opposite |
| Lag-SD value drives the logged growth | SD accounts for a substantial part of the validation increase | Excluded for the value being plotted: 98.84% of growth is VFM CE |
| Prefix-conditioned memorization | Seen blocks after block 1 become nearly perfect; block 1 remains hard; unseen train tail and validation fail similarly; wrong confidence rises | All predictions observed |

## Fixed-input discriminator

The probe uses 64 decorrelated windows per split and exactly the same cached
Gaussian `x0` and block times at every checkpoint. `train_unseen_tail` samples
the 89.65M characters of `train.bin` that the run never loads.

| Checkpoint | Seen train-prefix CE | Unseen train-tail CE | Validation CE |
|---:|---:|---:|---:|
| Source weights | 1.346 | 1.715 | 1.725 |
| 10k | 0.177 | 3.024 | 3.012 |
| 20k | 0.131 | 4.182 | 4.207 |
| 90k | 0.116 | 4.634 | 4.643 |

The unseen tail and validation track each other almost exactly. Therefore the
failure follows membership in the truncated training prefix, not whether text
came from `train.bin` or `val.bin`.

Evidence:

- [`fixed_probe.csv`](./fixed_probe.csv)
- [`fixed_probe_mechanism.json`](./fixed_probe_mechanism.json)
- [`fixed_probe_dynamics.png`](./fixed_probe_dynamics.png)

## Mechanistic localization by block

At step 90k:

- block 1 CE is `1.776` on the seen prefix, `1.704` on unseen train text, and
  `1.807` on validation;
- seen-prefix block 2 CE is `0.0795`;
- seen-prefix blocks 3–16 are almost all approximately `1e-6`;
- unseen/validation blocks 2–16 are approximately `3.3–5.6`.

Block 1 has no preceding clean block. Blocks 2–16 can attend to strictly
previous clean blocks. This is the distinctive signature of using the clean
prefix as a lookup key for a memorized continuation.

It also explains the apparent training plateau. Averaging the remaining
block-1 CE over 16 blocks gives `1.776/16 = 0.111`; adding the small block-2
residual yields the measured fixed-probe train CE `0.116`, close to the logged
training plateau.

The time-bin cut supports the same mechanism. At step 90k, unseen-tail CE is
`6.93`, `6.63`, `4.93`, and `0.37` across increasing `t` quartiles. Failure is
largest when the current block contains little target information and the
model must rely most heavily on the clean prefix.

Evidence:

- [`fixed_probe_by_block.csv`](./fixed_probe_by_block.csv)
- [`fixed_probe_details.json`](./fixed_probe_details.json)
- [`ce_by_block.png`](./ce_by_block.png)
- block mask definition: `block/mask.py`, lines 35–50

## Why cross-entropy rises rather than merely plateaus

On fixed validation inputs, from the source weights to step 90k:

- accuracy falls from `0.498` to `0.388`;
- mean maximum confidence rises from `0.490` to `0.807`;
- confidence on wrong predictions rises from `0.239` to `0.731`;
- predictive entropy falls from `1.759` to `0.577`.

The model is not merely failing to improve. It is sharpening around memorized
but incorrect continuations. True validation tokens receive progressively
smaller probabilities, so CE rises above the uniform 27-way value
`ln(27) = 3.296`.

## Excluded primary mechanisms

- Comet/TorchMetrics aggregation error.
- Validation RNG as the source of the main increase.
- A peculiar `val.bin` domain shift.
- NaN/Inf or global optimizer-state corruption.
- Dropout evaluation mode.
- Own-target attention leakage.
- Scheduler behavior: this run has no scheduler.
- ECLD/Triton behavior: this run uses Lag SD, and the measured rise is in the
  ordinary VFM CE.

## Remaining uncertainty

- A small Lag-SD scalar does not prove that its gradient is small. Its
  independent causal contribution needs a same-start `sd_prop=0` versus Lag
  ablation.
- Constant `lr=1e-4`, fresh AdamW, and no scheduler likely accelerate
  specialization. Relative parameter drift from the source is `19.6%` at 10k,
  `31.3%` at 30k, and `47.5%` at 90k. Their independent contribution needs a
  matched LR/optimizer ablation on corrected data.
- A full counterfactual correction still requires the same-start fine-tune on
  the complete Text8 corpus. That intervention would quantify how much of the
  failure remains intrinsic to the blockwise objective.
- The source CFM checkpoint was itself trained with the same truncated-loader
  settings, so a completely clean comparison ultimately requires retraining
  the source CFM on corrected data as well.

The supported causal conclusion for this run is nevertheless narrow and
strong: the observed validation growth is an overconfident
prefix-to-continuation memorization failure, enabled by the 256-times collapsed
training support.
