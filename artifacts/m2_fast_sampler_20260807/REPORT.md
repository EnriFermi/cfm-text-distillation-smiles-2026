# M2 rerun with the M3 maximum-speed sampler

Date: 2026-08-07. Device: NVIDIA H100 NVL. FP32, batch-one latency,
sequence length 256, block size 16, seed 0. Quality uses 512 argmax
generations per NFE.

## Failure definition

The previous plot reported M2 at 1,793.50 ms (NFE 16) and 3,538.57 ms
(NFE 32), versus M3-fast at 190.09 and 368.82 ms. Because M2 and M3 have the
same 384x6 architecture and differ primarily in checkpoint weights, this was
not a fair latency comparison.

## Validity checks

- Old M2 and M2-fast load the exact same step-68,001 checkpoint; the recorded
  checkpoint hash head is
  `4a1af3fe7008567a71338e138830b56015ade41bbdb165aba1611bfb9e3041b0`.
- M2-fast and M3-fast use the same `block_causal_fast` sampler, block size 16,
  FP32 dtype, current-block queries, finalized-prefix K/V cache, dynamic-prefix
  `torch.compile` graph, argmax decoding, and H100 NVL.
- Latency uses the timing function loaded verbatim by AST from upstream commit
  `2e21c57704ab94e131e3350ffd8286632f61a7b8`, with five warmups and 30
  repetitions.
- All six quality points were regenerated and rescored rather than inheriting
  the old sampler's MAUVE/gen-PPL.
- The targeted block suite passes 25/25 tests.

## Competing mechanisms and predictions

1. **Weights cause the latency gap.** If true, M2 should remain much slower
   than M3 after both use the same sampler.
2. **Sampler/backend mismatch causes the latency gap.** If true, M2-fast and
   M3-fast should have nearly identical latency at every NFE, while M2 quality
   remains poor.
3. **Fast inference changes the effective M2 model.** If true, the quality
   curve should move substantially rather than showing only small numerical
   variation.

## Results

| NFE | old M2 ms | fast M2 ms | speedup | fast M3 ms | fast M2 / M3 |
|---:|---:|---:|---:|---:|---:|
| 1  | 112.758 | 22.344 | 5.05x | 22.531 | 0.9917 |
| 2  | 222.505 | 33.403 | 6.66x | 33.610 | 0.9938 |
| 4  | 440.654 | 55.664 | 7.92x | 56.013 | 0.9938 |
| 8  | 882.778 | 100.151 | 8.81x | 100.713 | 0.9944 |
| 16 | 1793.504 | 189.186 | 9.48x | 190.093 | 0.9952 |
| 32 | 3538.574 | 367.033 | 9.64x | 368.817 | 0.9952 |

M2-fast is within 0.5--0.8% of M3-fast latency at every NFE. This excludes
checkpoint weights as the cause of the original order-of-magnitude gap and
supports the backend-mismatch mechanism.

Quality remains collapsed. M2-fast reaches MAUVE 0.01645 / gen-PPL 255.90 at
NFE 16 and MAUVE 0.02672 / gen-PPL 223.68 at NFE 32. The old values were
0.01457 / 256.13 and 0.02619 / 224.59, respectively. Token equality between
old and fast M2 ranges from 98.6% at NFE 1 to 96.3% at NFE 32; sequence-level
equality declines because small floating-point differences propagate through
later generated blocks. The quality shifts are small relative to M2's failure.

## Excluded and remaining mechanisms

- **Weights as a speed mechanism is excluded:** matched-backend latency is
  essentially identical.
- **A stale/wrong checkpoint is excluded:** path, step, and checkpoint hash
  match.
- **A major semantic change is not supported:** quality remains in the same
  collapsed regime and most individual tokens agree, although the SDPA/Flex
  paths are not bitwise identical.
- The quality failure itself remains attributable to the untrained clean-prefix
  conditioning behavior of M2; this rerun isolates inference speed, not that
  modeling failure.

## Narrow conclusion

The old M2-versus-M3 latency comparison was invalid. The approximately 10x gap
came from comparing the full doubled-stream sampler against the cached/compiled
current-block sampler. With the same maximum-speed backend, M2 and M3 cost the
same; their experimentally meaningful difference is generation quality.

## Evidence

- `comparison.csv`: old M2, M2-fast, and M3-fast latency at all NFE values.
- `old_vs_fast_token_parity.csv`: token and sequence equality fractions.
- `dump_fast_argmax.log`: sample-generation log.
- `score_fast_argmax.log`: MAUVE/gen-PPL scoring log.
- `latency_fast_argmax.log`: exact upstream-timer output.
- `regenerate_combined_plots.log`: updated combined-grid generation.
- `../../dumps/ts_M2_source_fast_cached_argmax.json` and `.tokens.npz`: 512
  generations per NFE.
- `../../results/ts_M2_source_fast_cached_argmax_scores.json`: six scored
  quality rows.
- `../tinystories_latency_quality_grid_20260806/latency_quality_grid.json`:
  66-row audit grid containing old and fast M2 alongside M3.
