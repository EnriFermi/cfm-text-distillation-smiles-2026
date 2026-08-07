# M3 inference causal profile (independent agent)

Date: 2026-08-06. GPU: NVIDIA H100 NVL. Checkpoint:
`logs/train/2026-07-28_12-43-24_12345_local/checkpoints/step_0100000.ckpt`
(`global_step=100000`). Batch 1, length 256, block size 16, vocabulary 50,257,
hidden 384, 6 layers, 6 heads, FP32.

## Failure definition

Current M3 block-causal inference is roughly 1.8 s at NFE=16, while full CFM
M1 is roughly 0.13 s at NFE=32. The issue to explain is M3's excess inference
latency, not its quality.

## Validity checks

- The profiled checkpoint and architecture match `dumps/ts_M3_step100000.json`.
- Current sampler is `block.sampling.block_causal_sample`.
- Each sequence has 16 blocks; current sampler invokes the doubled network
  `16 * NFE` times.
- Profile and benchmarks ran exclusively at batch 1 in FP32 on H100 NVL.
- One stable timing run measured 115.44--118.24 ms for NFE=1 (median 117.25
  ms). A separate one-forward run measured median 7.000 ms.

## Competing causal mechanisms and discriminators

1. **Whole doubled-sequence recomputation.** Every flow step for one 16-token
   block recomputes 512 transformer positions: all 256 clean positions and all
   256 noisy positions. Prediction: latency scales with `16*NFE`, and an
   incremental current-block forward with cached finalized-prefix K/V should
   reduce it substantially. Observed: one forward is 7.000 ms and NFE=1 is
   117.25 ms (16 forwards). A standalone incremental current-block + clean-KV
   prototype, compiled in reduce-overhead/CUDAGraph mode, reduced NFE=1 to
   38.64 ms (3.03x). The pre-existing argmax latency grid is almost perfectly
   linear in NFE: least-squares `latency_ms = 114.696*NFE - 0.523`,
   `R^2=0.999968` across NFE 1,2,4,8,16,32. This is the leading mechanism.

2. **Unused full-vocabulary projections and dense simplex allocation.** The
   current forward embeds and emits all 512 positions in 50,257 dimensions,
   but the sampler slices only 16 noisy output positions. Prediction: input and
   output vocabulary GEMMs and their tensors are prominent, and restricting
   them to the current block saves work/memory. Observed per-forward self-CUDA:
   input projection 337.5 us (17.0%), output projection 208.3 us (10.5%). Each
   dense input and output has 25,731,584 FP32 elements (~98.2 MiB), while only
   804,112 output elements are used (3.125%; 31/32 is discarded). The NFE=1
   profiler recorded 11.59 GiB of aggregate CUDA allocations. This contributes
   materially, but is not the sole mechanism.

3. **FlexAttention itself is pathological.** Prediction: attention kernels
   dominate one-forward GPU time. Observed compiled FlexAttention was 117.2 us
   across 6 layers (5.9% of self-CUDA kernel time). This excludes attention as
   the primary cause of the current 7 ms forward, although incremental
   attention remains necessary to avoid computing 512 queries.

4. **Softmax over 50k is the main cost.** Prediction: block endpoint softmax
   dominates sequence time. Observed NFE=1 softmax total was 314.2 us across 16
   calls (0.88% of self-CUDA time). The current code already slices to
   `(1,16,50257)` before softmax, so softmax is not the latency cause.

5. **Small-op/kernel-launch and allocation overhead.** Prediction: summed GPU
   kernels are much shorter than end-to-end forward latency; graph compilation
   helps more after work is reduced to a tiny block. Observed one-forward
   profiler summed 1.986 ms self-CUDA versus 7.000 ms end-to-end. Eager
   incremental execution was actually slower than current at NFE=1 (128.29 vs
   116.21 ms), while `torch.compile(mode="reduce-overhead")` with CUDAGraphs
   reduced it to 38.64 ms. Launch/allocator overhead is therefore a causal
   co-bottleneck and compile/graphs are required, not optional polish.

6. **FP32 arithmetic.** BF16 could reduce GEMM and memory costs, but H100's
   current FP32 matmuls already use TF32 kernels. BF16 is an approximate
   intervention and was not run after GPU exclusivity was requested. Its speed
   and MAUVE/genPPL parity remain not established.

## Prototype implementation

`prototype_incremental.py` implements, outside shared source:

- noisy transformer computation for only the current 16-token block;
- K/V caches for strictly previous finalized clean blocks at every layer;
- one clean-block cache update after discretizing a block (skipped after the
  final block);
- output projection and softmax only for the current noisy block;
- exact block-causal graph by presenting each query only the allowed keys;
- optional `torch.compile(fullgraph=True, mode="reduce-overhead")`, which uses
  CUDAGraphs here.

The change preserves the mathematical mask: noisy block `b` attends clean
blocks `<b` and its own noisy block; a finalized clean block attends clean
blocks `<=b`. It avoids future/junk noisy positions entirely.

The block graph itself was checked on a random small FP32 CPU BlockDIT using
dense SDPA on both paths: every block's logits matched bit-for-bit (`max_abs=0`)
and argmax agreement was 100% (`incremental_semantics_cpu.json`). Numerical/token
parity against the production CUDA **FlexAttention** path is still not
established. The prototype uses SDPA over the exact allowed K/V set, while the
baseline uses FlexAttention; floating-point reduction order may cause small
differences. Before merging, compare per-step logits/endpoints, argmax token
agreement under fixed prior draws, and rerun MAUVE/genPPL if any tokens differ.

## Narrow supported conclusion

The high M3 latency is caused primarily by an inference implementation that
reexecutes a training-oriented doubled 512-token network for each 16-token
block step and projects all 512 outputs to the vocabulary even though only 16
are consumed. Once transformer work is restricted to the current block and
clean K/V is cached, small-op launch overhead becomes dominant; compilation /
CUDA graphs is needed to realize the speedup. The measured FP32 NFE=1 prototype
speedup is at least 3.03x in the first run. A later grid was contaminated by a
concurrent BF16 GPU run and contains impossible NFE16/NFE32 equality; none of
its latency rows are valid final evidence. NFE=2--32 scaling and production
FlexAttention numerical/quality parity still require an exclusive benchmark.

## Artifacts

- `current_profile_summary.json`: checkpoint, dimensions, raw timings.
- `current_forward_table.txt`, `current_forward_trace.json`: one forward.
- `current_sequence_nfe1_table.txt`, `current_sequence_nfe1_trace.json`: full
  NFE=1 sequence and allocation evidence.
- `prototype_incremental.py`: standalone implementation.
- `check_incremental_semantics_cpu.py`, `incremental_semantics_cpu.json`: exact
  graph discriminator against the full doubled dense-SDPA path.
- `prototype_benchmark.json`: latest contaminated grid (retained for audit,
  explicitly invalid as final latency evidence).
- `profile_current.log`, `prototype_incremental_smoke.log`,
  `prototype_compiled_smoke2.log`: run logs.
