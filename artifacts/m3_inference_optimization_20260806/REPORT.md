# M3 maximum-speed inference optimization

Date: 2026-08-06. Device: NVIDIA H100 NVL. FP32, batch 1, sequence length
256, block size 16. Checkpoint:
`logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt`
(`global_step=100000`, SHA-256 head
`1a69c085a780f9d3af00e1e3a923611c7c6b65ccb85c7eac6cff36be51cdc7f5`).

## Failure definition

The original M3 sampler takes 1.830 s at argmax NFE=16 and 3.673 s at
argmax NFE=32. The goal was maximum steady-state batch-1 inference speed for
the existing checkpoint, while measuring any quality change instead of
assuming numerical parity.

## Validity checks

- Old and fast rows use the same checkpoint path, global step, checkpoint hash,
  length, block size, FP32 dtype, seed and evaluation samples count.
- Latency uses warmup=5 and repeats=30. The `_sync` and
  `_measure_sequence_latency_ms` AST nodes are loaded directly from upstream
  commit `2e21c57704ab94e131e3350ffd8286632f61a7b8`; the copied source and its
  hash are stored beside the results.
- Final latency ran with no other GPU compute process. The earlier contaminated
  prototype grid is retained for audit but excluded from every conclusion.
- Quality was recomputed on 512 newly generated texts per point, using the same
  2,048-text TinyStories reference, `gpt2-large` MAUVE features and FP32
  `EleutherAI/gpt-j-6B` gen-PPL protocol as the original M3 grid.
- The combined result contains 60/60 unique model/point rows and no NaN/inf.
- The exact mask algebra was checked on CPU, where full doubled dense-SDPA and
  incremental dense-SDPA agree. The targeted block suite passes 25/25 tests.

## Competing mechanisms and predictions

1. **Training-oriented full doubled-stream recomputation.** The old sampler
   rebuilds 256 clean + 256 noisy positions for every one of `16 * NFE` flow
   jumps. If this dominates, latency should be almost linear in NFE and
   restricting queries to the current 16-token block while caching finalized
   clean K/V should provide the main speedup.

2. **Discarded vocabulary projections.** The old network projects all 512
   positions from/to vocabulary size 50,257 and consumes only 16 output rows.
   If material, profiling should show the two vocabulary GEMMs and large
   allocations, and current-block projection should reduce both.

3. **Kernel-launch / allocator overhead after reducing the graph.** If the
   incremental eager graph consists of too many small operations, its summed
   CUDA kernels should be much shorter than wall time and eager caching alone
   should fail to realize the theoretical gain. Compiling the flow update and
   clean encoder into CUDA graphs should then be necessary.

4. **FlexAttention is the primary bottleneck.** If true, attention kernels
   should dominate self-CUDA time and KV caching without replacing attention
   should yield limited gains.

5. **The 50k softmax is the primary bottleneck.** If true, current endpoint
   softmax should dominate sequence time and transformer/KV work reduction
   should yield limited gains.

6. **Reduced precision is the main remaining opportunity.** If true, BF16
   should beat FP32 on H100 without unacceptable trajectory changes.

## Discriminating measurements and results

The old path executes exactly 16 doubled forwards at NFE=1. One forward has a
7.000 ms median, while the sequence has a 117.25 ms median in the isolated
profile. Across the prior final grid, argmax latency fits
`114.696 * NFE - 0.523 ms` with `R^2=0.999968`. This supports mechanism 1.

Per old forward, input and output vocabulary projections account for 17.0%
and 10.5% of self-CUDA respectively. The input and output each contain
25,731,584 FP32 elements (98.2 MiB), but only 3.125% of output logits are used;
the NFE=1 sequence profiler recorded 11.59 GiB aggregate CUDA allocations.
This supports mechanism 2 as a material contributor.

Attention accounts for only 5.9% of old-forward self-CUDA and endpoint softmax
for 0.88% of old NFE=1 self-CUDA. These observations exclude mechanisms 4 and
5 as primary causes.

An eager current-block/KV path remained launch-bound: the profiler contained
1,241 linears, 186 SDPA calls and roughly 1,800 elementwise multiply launches
for NFE=1. Compiling dynamic-prefix flow steps and clean-block encoders with
`torch.compile(fullgraph=True, mode="reduce-overhead")`, while keeping softmax
and the Euler update inside the compiled region, realizes the gain. This
supports mechanism 3.

BF16 was slower in the exclusive diagnostic (for example fixed cached NFE=16:
451.6 ms versus 276.5 ms FP32) and produced more trajectory divergence. It is
therefore excluded for this H100/checkpoint path; H100 already uses TF32 for
the relevant FP32 matmuls.

### Final exact-upstream latency

| mode | NFE | old M3 ms | fast M3 ms | speedup |
|---|---:|---:|---:|---:|
| argmax | 1 | 112.548 | 22.531 | 5.00x |
| argmax | 2 | 242.929 | 33.610 | 7.23x |
| argmax | 4 | 451.758 | 56.013 | 8.07x |
| argmax | 8 | 911.942 | 100.713 | 9.06x |
| argmax | 16 | 1830.439 | 190.093 | 9.63x |
| argmax | 32 | 3673.105 | 368.817 | 9.96x |
| sample | 1 | 132.985 | 23.708 | 5.61x |
| sample | 2 | 261.398 | 34.672 | 7.54x |
| sample | 4 | 517.837 | 57.008 | 9.08x |
| sample | 8 | 1031.344 | 101.705 | 10.14x |
| sample | 16 | 1849.754 | 191.188 | 9.68x |
| sample | 32 | 3594.005 | 369.845 | 9.72x |

The first fast latency point took 22.2 s wall time because Inductor compiled
the 16 prefix-length specializations. Reported sequence latency is steady-state
after five warmups; production should prewarm the required batch/NFE shape.

### Quality re-evaluation

| mode | NFE | old MAUVE | fast MAUVE | old gen-PPL | fast gen-PPL |
|---|---:|---:|---:|---:|---:|
| argmax | 8 | 0.1570 | 0.2821 | 50.775 | 50.565 |
| argmax | 16 | 0.7979 | 0.7550 | 59.086 | 58.128 |
| argmax | 32 | 0.6271 | 0.6443 | 61.887 | 61.840 |
| sample | 8 | 0.2882 | 0.3473 | 61.211 | 61.271 |
| sample | 16 | 0.5978 | 0.5019 | 70.729 | 70.744 |
| sample | 32 | 0.5777 | 0.6461 | 76.178 | 75.278 |

Gen-PPL is stable across the important points. MAUVE is not uniformly ordered:
fast is higher at NFE=8 and NFE=32, but lower at NFE=16, most visibly for the
sample decoder. This is not bitwise-preserving inference. The graph/mask is the
same, but the old CUDA path uses FlexAttention over the doubled layout while
the fast path uses dense SDPA over only the allowed keys; reduction-order
differences perturb some token decisions and then autoregress through later
blocks. Consequently the newly scored fast metrics, not the old metrics, are
the valid quality evidence for the fast curve.

## Implementation

- `block/fast_inference.py`: current-block execution, incremental finalized
  clean-prefix K/V, dynamic-prefix compiled flow step, compiled clean encoder,
  and output-buffer ownership handling for CUDA graph replay.
- `block/sampling.py`: explicit `compiled_cached` backend; the default `full`
  backend remains unchanged for reproducibility.
- `eval/dump_samples.py`: explicit `block_causal_fast` sampler for quality
  generation.
- `eval/measure_quality_grid_latency.py`: fast M3 series in the exact-upstream
  latency/quality grid.
- `tests/test_block.py`: CPU full-versus-cached token equivalence over argmax /
  sample and NFE 1/2/4.

## Excluded and remaining mechanisms

- FlexAttention and endpoint softmax are measured minor costs, not causes of
  the original latency.
- BF16 is slower here and is not used.
- Stale checkpoint, split, sampler label and metric reuse are excluded by the
  matching hashes/metadata and new text dump/score file.
- The remaining irreducible checkpoint cost is the dense simplex state: every
  flow jump still applies the 50,257-to-384 input projection and
  384-to-50,257 output projection for the current 16 tokens. Removing that
  requires a different training/state parameterization, not a checkpoint-only
  inference optimization.
- A strictly bitwise FlexAttention-compatible fast tier is not established.
  The implemented tier prioritizes the requested maximum speed and reports its
  own measured quality.

## Narrow supported conclusion

M3's excessive latency was caused primarily by recomputing and projecting the
entire doubled 512-position training graph for every 16-token block jump, with
kernel-launch overhead becoming a co-bottleneck after graph reduction. The
implemented current-block K/V-cached compiled sampler cuts steady-state
batch-1 latency by 9.6--10.0x at NFE 16--32. It preserves gen-PPL and broadly
preserves MAUVE, but is a numerically approximate maximum-speed tier rather
than a bitwise-identical replacement.

## Evidence artifacts

- `artifacts/tinystories_latency_quality_grid_20260806/latency_quality_grid.json`
  and `.csv`: final 60-row quality/latency grid.
- `artifacts/tinystories_latency_quality_grid_20260806/mauve_vs_latency.png`
  and `genppl_vs_latency.png`: inspected final plots.
- `results/ts_M3_step100000_fast_cached_scores.json`: all 12 new quality rows.
- `dumps/ts_M3_step100000_fast_cached.json` and `.tokens.npz`: newly generated
  texts and token IDs.
- `artifacts/m3_inference_optimization_20260806/profile_agent/REPORT.md` and
  profiler traces/tables: causal profile.
- `artifacts/m3_inference_optimization_20260806/dump_fast.log`,
  `score_fast.log`, `latency_fast.log`: runtime logs.
- `artifacts/m3_inference_optimization_20260806/profile_agent/prototype_dynamic_exclusive_run.log`:
  independent exclusive prototype confirmation.
