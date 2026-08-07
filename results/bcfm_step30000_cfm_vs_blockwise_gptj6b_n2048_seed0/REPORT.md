# CFM vs Blockwise CFM PPL — BCFM checkpoint step 30,000

## Protocol

- Checkpoint: `step_0030000.ckpt`, global step 30,000, epoch 10.
- Checkpoint SHA-256:
  `fa7dc9859c595584e73f88a5464108af897cc1cce4ae835e3741d77583df3c7d`.
- Same network weights in both formulations:
  - `full_cfm`: weights strict-loaded into the original full-sequence Duo `DIT`;
  - `blockwise_cfm`: weights strict-loaded into `BlockDIT` and sampled with the
    trained block-causal mask at block size 16.
- 2,048 generated Text8 sequences, length 256, seed 0, argmax decoding.
- Sampling batch size 128; synchronous CPU transfer (`non_blocking=False`).
- GPT-J-6B judge in FP32, GPT-2-large tokenizer, judge batch size 4.
- NFE values: 1, 2, 4, 8.

For full CFM, NFE is the number of full-sequence steps. For Blockwise CFM, it is
the number of steps per block. With 16 blocks, the latter therefore performs
16, 32, 64, or 128 model calls per generated sequence.

## Results

| Formulation | NFE | Total model calls | Mean NLL | PPL | Mean per-sample entropy, B=16 |
|---|---:|---:|---:|---:|---:|
| Full CFM | 1 | 1 | 7.195726 | 1333.718 | 2.2029 |
| Full CFM | 2 | 2 | 7.326822 | 1520.542 | 2.2136 |
| Full CFM | 4 | 4 | 7.359238 | 1570.639 | 2.2117 |
| Full CFM | 8 | 8 | 7.390564 | 1620.620 | 2.2183 |
| Blockwise CFM | 1 | 16 | 6.958360 | 1051.907 | 2.1624 |
| Blockwise CFM | 2 | 32 | 6.975264 | 1069.839 | 2.1733 |
| Blockwise CFM | 4 | 64 | 6.947032 | 1040.058 | 2.1853 |
| Blockwise CFM | 8 | 128 | 6.859085 | **952.495** | 2.1917 |

## Review

- Best full-sequence result is NFE=1. Increasing full CFM NFE monotonically
  worsens PPL for this block-fine-tuned checkpoint.
- Blockwise is better at every equal nominal step count, by 21.1%, 29.6%,
  33.8%, and 41.2%, respectively. This is not a compute-matched comparison:
  Blockwise has 16 times as many model calls at the same displayed NFE.
- The Blockwise 1/2/4 differences are small and non-monotonic. The clearest
  improvement is at NFE=8. This sweep has one generation seed and does not
  establish confidence intervals for the small differences.
- There is no evidence of trivial entropy or repetition collapse:
  all 2,048 sequences at every point are unique; token IDs are valid 0–26;
  maximum identical-character runs are only 4–6 characters.
- For reference, the existing real-Text8 sample has mean per-sample B=16
  entropy 2.2319 nats. Full CFM spans 2.2029–2.2183 and Blockwise spans
  2.1624–2.1917. Blockwise is somewhat less diverse, but its entropy increases
  rather than decreases as PPL improves toward NFE=8.
- A mild late-block degradation remains. At Blockwise NFE=8, B=16
  per-sample entropy changes from 2.2432 on the first block to 2.1938 on the
  final block. This is consistent with modest accumulated prefix/exposure
  error, not collapse.
- Sampling throughput in `summary.json` is diagnostic only: training was
  running on the same GPU. In addition, the current `block_causal_sample`
  recomputes the doubled masked stream for every block and does not use a
  persistent inference KV cache. This does not invalidate PPL, but it is not a
  production speed measurement of an optimized cached Blockwise sampler.

## Artifacts

- `nll_vs_nfe.png`: NLL curves for both formulations.
- `summary.json` / `summary.csv`: primary PPL, NLL, entropy, and NFE table.
- `protocol.json`: resolved model configs, checkpoint hash, device, dtype, seed,
  batch sizes, and transfer mode.
- `review.json`: tensor validity, uniqueness, repetition checks, GPT-token
  lengths, and per-block entropy curves.
- `*_tokens.pt`: all generated token tensors.
- `*_samples.txt`: all generated strings.
- `scripts/compare_cfm_bcfm_ppl.py`: reproducible evaluator.
