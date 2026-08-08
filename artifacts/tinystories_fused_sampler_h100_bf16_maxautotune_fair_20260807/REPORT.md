# M2/M3/BD3-LM inference optimality and fairness audit

## Question and success criterion

Question: was the previously plotted M2/M3 fast path already optimal, and were
the low-level accelerations fair relative to BD3-LM?

Success requires one matched batch-1 H100 protocol, the same generally
applicable precision and compiler policy for all three models, architecture-
specific caching only where it preserves each sampler, and quality rescoring
whenever the numerical path changes.

## Narrow answer

No. The previous path was a strong FP32-tensor/TF32-GEMM PyTorch baseline, but
it was not maximum speed. A shared BF16 policy plus a shared TorchInductor
`max-autotune` policy reduces M2/M3 latency substantially. The same policies
were applied to BD3-LM; their smaller effect there is attributable to a
different operation mix, not a different benchmark protocol.

These new latencies must **not** yet replace the quality-latency curves: BF16
and autotuned GEMM reduction order change the numerical path, so M2, M3, and
BD3-LM generations must be regenerated and rescored before pairing MAUVE or
genPPL with these measurements.

## Matched protocol

- GPU: NVIDIA H100 NVL
- batch size: 1
- sequence length: 256
- block size: 16
- warmups: 5
- measured repetitions: 30
- statistic: median, with p10 and p90 retained
- timing implementation: AST-loaded directly from upstream commit
  `2e21c57704ab94e131e3350ffd8286632f61a7b8`
- shared fast-path features: current-block execution, compiled CUDA graphs,
  fused clean-previous/current-first transition, no separate cache-maintenance
  forwards, no diagnostic CPU synchronization
- shared tested compute policies:
  1. FP32 tensors with TF32 GEMMs, `reduce-overhead`
  2. BF16 parameters/compute, `reduce-overhead`
  3. BF16 parameters/compute, `max-autotune`

M2, M3, and BD3-LM all have six Transformer layers, hidden size 384, six heads,
head dimension 64, block size 16, and an approximately 50k-token vocabulary.
Their algorithmic differences remain intact: CFM evolves a dense simplex state
and uses argmax endpoints; BD3-LM uses token embeddings and canonical top-p
stochastic sampling.

## Reviewed latency results

Median sequence latency in milliseconds:

| model / point | prior FP32+TF32 | shared BF16 | BF16 + shared max-autotune | total speedup |
|---|---:|---:|---:|---:|
| M2, steps/block 1 | 16.388 | 13.543 | 12.531 | 1.308x |
| M2, steps/block 16 | 184.418 | 149.005 | 114.186 | 1.615x |
| M2, steps/block 32 | 363.091 | 293.345 | 224.213 | 1.619x |
| M3, steps/block 1 | 16.519 | 13.504 | 13.261 | 1.246x |
| M3, steps/block 16 | 184.201 | 149.294 | 114.258 | 1.612x |
| M3, steps/block 32 | 362.635 | 293.833 | 224.127 | 1.618x |
| BD3-LM, steps/block 1 | 21.393 | 20.088 | 19.612 | 1.091x |
| BD3-LM, steps/block 16 | 257.515 | 215.042 | 212.166 | 1.214x |
| BD3-LM, steps/block 32 | 509.755 | 420.658 | 416.538 | 1.224x |
| BD3-LM first-hitting | 269.009 | 259.749 | 266.511 | 1.009x |

The complete 19-point comparison is in
`precision_compile_comparison.csv`; all 31 raw rows, including CFM sample
decoding rows, are in `fused_sampler_latency.csv` and
`fused_sampler_latency.json`.

One raw M2 steps/block=4 run has a wide p10-p90 interval (33.282-50.174 ms).
The neighboring points and the M3 matched-architecture measurement do not show
the same anomaly. It does not affect the main steps/block=16 or 32 conclusion,
but that row should be repeated before publication-quality error bars are used.

## Competing bottlenecks and discriminating evidence

### 1. Precision/GEMM throughput

Prediction: a shared BF16 policy should lower the per-backbone-call slope for
all models, especially where matrix multiplication dominates.

Result: M2/M3 improve by about 1.21-1.24x under BF16 alone across the full NFE
sweep. BD3-LM improves from 1.06x at one step/block to 1.21x at 32 steps/block.
This mechanism is supported. The smaller BD3 low-NFE win is consistent with
fixed top-p sampling and mask-update costs occupying a larger share of runtime.

### 2. Compiler kernel selection

Prediction: `max-autotune` should help most when the default kernel is poor for
a large, unusual GEMM shape.

Result: it gives another roughly 1.30x at M2/M3 steps/block 16-32, but only
about 1.01x at BD3 steps/block 16-32. Autotune logs identify CFM's dense
simplex-to-hidden projection with shape `(16 x 50257) @ (50257 x 384)` as the
operation with the unusual large reduction dimension. BD3 uses a token-
embedding lookup at input and therefore does not contain that GEMM. The policy
was identical; the operation mix explains the differential benefit.

### 3. KV concatenation / attention backend

Prediction: a true preallocated FlashAttention KV-cache kernel should beat the
current `torch.cat + SDPA` attention subpath, increasingly with prefix length.

Result: an isolated BF16 CUDA-graph microbenchmark shows FA2 KV-cache speedups
of 1.24x, 1.31x, 1.51x, and 1.52x at prefix lengths 0, 64, 128, and 240. The
artifact is `attention_backend_microbench.json`.

This does **not** establish the same end-to-end speedup. Attention is only part
of a backbone call. Integrating the installed FA2 mutation kernel as a traced
custom op causes TorchInductor to disable CUDA Graph capture because K/V inputs
are mutated. The prior literal preallocated-cache implementation similarly took
31.321 ms versus 16.029 ms for dynamic fused KV at M3 steps/block=1. Therefore
FA2 KV-cache is a remaining custom-kernel opportunity, not a completed win.

### 4. Cache-maintenance forwards and CPU synchronization

These are excluded as current bottlenecks. Both CFM and BD3-LM make exactly
`16 * steps_per_block` backbone calls for their ordinary block sweeps, fuse the
previous clean-block commit with the next block's first useful forward, skip the
last commit, and keep the sampling loop on GPU. There are no separate cache
forwards in the measured rows.

## Validity findings

The original artifact recorded `float32_matmul_precision=highest` and
`allow_tf32=false` before model construction. Semicat construction then changed
the actual runtime to `high` and `allow_tf32=true`. The stored probe
`runtime_precision_probe.json` demonstrates this transition. Thus the old data
were not strict FP32 GEMMs; they were FP32 tensors with TF32 GEMMs. The new
runner now sets and records the shared policy before any model is built.

All 31 new rows passed schema/range checks, use the same H100, and contain 30
repetitions. CFM block tests pass (31 tests). The selected BD3 cache/parity tests
pass (4 tests); an initial invocation from the parent repository failed only
because its relative smoke-config paths require the external baseline directory
as the working directory.

## Remaining gap

The best fair measured path here is BF16 + shared `max-autotune`, not a claim of
global hardware optimality. A custom functional/static KV attention kernel that
preserves CUDA Graph capture remains viable. More importantly, MAUVE/genPPL for
the exact BF16+autotuned generations are not yet established, so the previous
quality-latency graph must remain unchanged until rescoring is complete.

## Artifacts

- `fused_sampler_latency.json`: full resolved config and raw measurements
- `fused_sampler_latency.csv`: raw measurements in tabular form
- `precision_compile_comparison.csv`: joined old/shared-BF16/max-autotune rows
- `attention_backend_microbench.json`: isolated SDPA-cat versus FA2 KV-cache
- `runtime_precision_probe.json`: original metadata timing issue

Source comparison artifacts:

- `../tinystories_fused_sampler_h100_20260807/fused_sampler_latency.json`
- `../tinystories_fused_sampler_h100_bf16_fair_20260807/fused_sampler_latency.json`

Reproduction command from the repository root:

```bash
/home/coder/.conda/envs/semicat/bin/python \
  -m eval.measure_fused_sampler_h100_latency \
  --precision bf16 \
  --compile-mode max-autotune \
  --output-dir artifacts/tinystories_fused_sampler_h100_bf16_maxautotune_fair_20260807 \
  --no-resume
```
