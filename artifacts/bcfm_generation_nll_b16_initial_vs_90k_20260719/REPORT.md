# B=16 generative NLL: source CFM weights versus 90k BCFM

Date: 2026-07-19

## Protocol

- Source: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/baseline/s_baseline.ckpt` (raw fine-tune initialization;
  checkpoint metadata step 390000).
- Tuned: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/logs/train/2026-07-18_10-46-57_12345_local/checkpoints/step_0090000.ckpt`.
- Block size: 16.
- NFE: 8, 16, 32 steps per block, corresponding to 128, 256, 512 model
  forwards per sequence.
- 2,048 generated sequences per point; 256 Text8 characters; seed 0; argmax
  discretization; sampling batch 128.
- Judge: GPT-J-6B; tokenizer: GPT-2-large; token-weighted mean NLL;
  PPL = exp(NLL).
- CPU transfer was synchronous (`non_blocking=False`).
- Uncertainty: 20,000 paired sequence-level bootstrap replicates using the
  same prior indices.

## Aggregate result

| NFE | Source NLL | 90k NLL | Source PPL | 90k PPL | Overall ΔNLL [95% CI] | ΔNLL after excluding 90k outputs with ≥64-char seen-train match [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.746890 | 6.593112 | 851.41 | 730.05 | -0.153781 [-0.180105, -0.128494] | +0.009079 [-0.008715, +0.026791] |
| 16 | 6.759532 | 6.582149 | 862.24 | 722.09 | -0.177389 [-0.206245, -0.148997] | +0.027924 [+0.009221, +0.046558] |
| 32 | 6.770277 | 6.577962 | 871.55 | 719.07 | -0.192316 [-0.222365, -0.162557] | +0.024741 [+0.005054, +0.044689] |

The aggregate judge metric improves significantly at every NFE. PPL falls by
14.3%, 16.3%, and 17.5% for NFE 8, 16, and 32.

## Memorization audit

All 2,048 sequences are unique at every point, and the 90k pooled character
entropy is slightly higher, so this is not a coarse duplicate or character
mode collapse.

However, exact suffix-automaton matching against the only 351,563 training
characters exposed by the loader shows:

| NFE | Source fraction with ≥64-char exact train match | 90k fraction with ≥64-char exact train match | 90k fraction with ≥128-char match | 90k exact 256-char train windows |
|---:|---:|---:|---:|---:|
| 8 | 0.0% | 13.92% | 13.43% | 1.37% |
| 16 | 0.0% | 16.65% | 16.11% | 1.81% |
| 32 | 0.0% | 17.43% | 16.70% | 1.90% |

At NFE 32, the non-copy stratum (`<64` matching characters) contains 82.6% of
sequences and 89.4% of judged GPT tokens. Within this stratum, NLL is 6.7963
for 90k versus 6.7716 for the source outputs at the same prior indices:
ΔNLL = +0.0247, i.e. slightly worse. The aggregate gain comes from the
roughly 16.7% long-copy subgroup, whose NLL is around 4.7 rather than 6.8.

## Conclusion

If the sole metric is aggregate GPT-J generative NLL, the 90k fine-tune
improved it decisively.

It is not evidence of improved generalization or novel-generation quality.
The gain is carried by long or exact copies from the truncated train prefix.
After excluding those copies, NLL is unchanged at NFE 8 and significantly
worse at NFE 16 and 32. The supported interpretation is memorization-driven
metric improvement, not a clean generation-quality improvement.

## Evidence

- `comparison.csv`: aggregate NLL/PPL and entropy.
- `paired_bootstrap.csv`: overall paired uncertainty.
- `stratified_bootstrap.csv`: copy-stratified paired uncertainty.
- `memorization_audit.csv`: exact train-substring audit.
- `copy_stratified_nll.csv`: NLL by copy length.
- `per_sample_judge.npz`: per-sequence NLL sums/token counts.
- `memorization_per_sample.npz`: per-sequence exact-match lengths.
- `top_seen_train_matches.json`: inspected examples.
- `final_comparison.png`: reviewed comparison plot.
- `source_n2048_seed0/` and `step90k_n2048_seed0/`: exact protocols,
  generated tokens, and generated strings.
