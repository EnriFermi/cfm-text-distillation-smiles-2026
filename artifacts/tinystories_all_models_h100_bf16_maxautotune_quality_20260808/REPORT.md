# Matched BF16/max-autotune M1 and MDLM results

Every listed point uses H100 NVL batch-one latency (5 warmups, median of 30 repeats) and newly generated BF16 quality samples (n=512).

| model | point | latency ms | MAUVE | gen-PPL |
|---|---:|---:|---:|---:|
| M1-200k | argmax_nfe1 | 1.033 | 0.005704 | 97.292 |
| M1-200k | argmax_nfe2 | 1.634 | 0.008250 | 83.563 |
| M1-200k | argmax_nfe4 | 2.818 | 0.021334 | 70.521 |
| M1-200k | argmax_nfe8 | 5.208 | 0.116346 | 69.146 |
| M1-200k | argmax_nfe16 | 9.951 | 0.546179 | 66.787 |
| M1-200k | argmax_nfe32 | 19.462 | 0.668426 | 67.868 |
| M1-68k | argmax_nfe1 | 1.018 | 0.005825 | 86.856 |
| M1-68k | argmax_nfe2 | 1.595 | 0.007168 | 82.374 |
| M1-68k | argmax_nfe4 | 2.816 | 0.010515 | 74.270 |
| M1-68k | argmax_nfe8 | 5.223 | 0.032873 | 77.858 |
| M1-68k | argmax_nfe16 | 9.935 | 0.220040 | 88.685 |
| M1-68k | argmax_nfe32 | 19.407 | 0.533137 | 89.951 |
| MDLM | sample_nfe16 | 32.183 | 0.897647 | 26.524 |
| MDLM | sample_nfe32 | 62.301 | 0.922320 | 19.892 |
| MDLM | sample_nfe64 | 121.547 | 0.903120 | 17.346 |
| MDLM | sample_nfe128 | 232.663 | 0.955582 | 16.080 |
| MDLM | sample_nfe256 | 442.778 | 0.918667 | 15.380 |

M1/MDLM data: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_all_models_h100_bf16_maxautotune_quality_20260808/m1_mdlm_optimized_quality_latency.csv`
All-model data: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_all_models_h100_bf16_maxautotune_quality_20260808/all_models_optimized_quality_latency.csv`
Metadata: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_all_models_h100_bf16_maxautotune_quality_20260808/metadata.json`
