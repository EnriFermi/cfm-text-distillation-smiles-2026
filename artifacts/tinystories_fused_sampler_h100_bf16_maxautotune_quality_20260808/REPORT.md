# BF16 max-autotune quality/latency join

All M2, M3, and BD3-LM rows use newly generated BF16 samples and matched BF16/max-autotune latency on the same H100 NVL.

| model | point | latency ms | MAUVE | gen-PPL |
|---|---:|---:|---:|---:|
| BD3-LM | sample_nfe1 | 19.612 | 0.004760 | 633.990 |
| BD3-LM | sample_nfe2 | 33.588 | 0.007656 | 238.392 |
| BD3-LM | sample_nfe4 | 60.141 | 0.279648 | 72.078 |
| BD3-LM | sample_nfe8 | 111.080 | 0.925242 | 33.877 |
| BD3-LM | sample_nfe16 | 212.166 | 0.889947 | 21.611 |
| BD3-LM | sample_nfe256 | 266.511 | 0.945909 | 13.520 |
| BD3-LM | sample_nfe32 | 416.538 | 0.940146 | 17.458 |
| M2 | argmax_nfe1 | 12.531 | 0.004530 | 51.802 |
| M2 | argmax_nfe2 | 21.849 | 0.005143 | 61.225 |
| M2 | argmax_nfe4 | 37.251 | 0.005512 | 149.974 |
| M2 | argmax_nfe8 | 63.290 | 0.005996 | 254.763 |
| M2 | argmax_nfe16 | 114.186 | 0.012915 | 253.942 |
| M2 | argmax_nfe32 | 224.213 | 0.021817 | 221.963 |
| M3 | argmax_nfe1 | 13.261 | 0.007826 | 48.755 |
| M3 | argmax_nfe2 | 21.530 | 0.010017 | 49.278 |
| M3 | argmax_nfe4 | 34.349 | 0.052039 | 46.356 |
| M3 | argmax_nfe8 | 60.747 | 0.455810 | 50.613 |
| M3 | argmax_nfe16 | 114.258 | 0.644107 | 58.278 |
| M3 | argmax_nfe32 | 224.127 | 0.722992 | 62.304 |

Joined data: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_bf16_maxautotune_quality_20260808/optimized_quality_latency.csv`
Metadata: `/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026/artifacts/tinystories_fused_sampler_h100_bf16_maxautotune_quality_20260808/metadata.json`
