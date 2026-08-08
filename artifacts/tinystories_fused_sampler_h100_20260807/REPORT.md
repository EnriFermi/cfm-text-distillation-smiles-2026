# Fused sampler H100 measurements

Protocol: NVIDIA H100 NVL, FP32, batch 1, length 256, block size 16, 5 warmups + 30 measured repeats; timing functions loaded from upstream commit `2e21c57704ab94e131e3350ffd8286632f61a7b8`.

## Median sequence latency (ms)

| steps/block | BD3-LM | M2 | M3 |
|---:|---:|---:|---:|
| 1 | 21.393 | 16.388 | 16.519 |
| 2 | 36.657 | 27.677 | 27.736 |
| 4 | 68.243 | 50.092 | 50.166 |
| 8 | 131.364 | 95.046 | 94.858 |
| 16 | 257.515 | 184.418 | 184.201 |
| 32 | 509.755 | 363.091 | 362.635 |

BD3-LM first-hitting (256 total denoising NFEs): 269.009 ms.

## Review notes

- Fusing the clean-prefix commit with the next block's first denoising call removes all separate cache-maintenance forwards.
- The optimized paths make exactly `16 × steps/block` backbone calls for ancestral/flow sampling; first-hitting makes 256 calls.
- Literal in-place preallocated KV was not selected: PyTorch reported that mutated inputs disabled CUDA Graph capture; at M3 NFE/block=1 it took 31.321 ms vs 16.029 ms for dynamic fused KV (1.954× slower).
- Therefore the fastest measured path still uses dynamic K/V concatenation with PyTorch SDPA. A true FlashAttention KV-cache kernel is not claimed by these measurements.
- M2/M3 quality was rescored because compiled SDPA ordering introduces small FP differences. BD3-LM quality is reused only after exact canonical-vs-fast token parity checks.
- Across M2/M3 argmax dumps, token agreement with the previous fast path is 96.579%–99.648%; the largest absolute MAUVE change is 0.0601. These are real numerical-path differences, so the new latency is paired only with the rescored quality.
- BD3-LM parity is exact at all seven measured points (seed 0, batch 1), stored in `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/bd3_token_parity.json`.
- MAUVE and genPPL use the existing 512-sample canonical protocol (except the separately identified 2048-sample BD3 headline, which is not used here).

## Artifacts

- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/optimized_quality_latency.csv`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/m2_m3_quality_deltas.csv`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/m2_m3_token_drift.csv`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/cache_backend_ablation.json`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/bd3_token_parity.json`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/mauve_vs_optimized_latency.png`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/genppl_vs_optimized_latency.png`
- `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_20260807/speedup_vs_previous_fast.png`
