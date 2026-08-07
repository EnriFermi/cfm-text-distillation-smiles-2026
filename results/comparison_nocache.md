# CFM vs BD3-LM generation latency (TinyStories, L=256)

- GPU: **NVIDIA GeForce RTX 5070 Laptop GPU**, torch 2.7.1+cu128, fp32
- CFM: `block_causal`, block_size 16, 51,957,969 params, step 100000
- BD3-LM: block_size 16, 51,119,442 params, ema weights, **KV cache OFF**
- CFM has no KV cache in any mode: `block_causal_sample` rebuilds the full `[clean; noisy]` 2L=512 forward on every flow-map jump.
- Every number is the median of 5 timed runs after 2 warmups, CUDA-synchronised.

## CFM

| batch | NFE (steps/block) | forwards/seq | s/batch | ms/seq | tok/s |
|---|---|---|---|---|---|
| 1 | 1 | 16 | 0.1082 | 108.2 | 2366 |
| 1 | 2 | 32 | 0.2163 | 216.3 | 1184 |
| 1 | 4 | 64 | 0.4356 | 435.6 | 588 |
| 1 | 8 | 128 | 0.8762 | 876.2 | 292 |
| 1 | 16 | 256 | 1.7784 | 1778.4 | 144 |
| 4 | 1 | 16 | 0.4232 | 105.8 | 2420 |
| 4 | 2 | 32 | 0.8441 | 211.0 | 1213 |
| 4 | 4 | 64 | 1.6865 | 421.6 | 607 |
| 4 | 8 | 128 | 3.3432 | 835.8 | 306 |
| 4 | 16 | 256 | 6.6864 | 1671.6 | 153 |
| 16 | 1 | 16 | 1.7127 | 107.0 | 2392 |
| 16 | 2 | 32 | 3.4264 | 214.2 | 1195 |
| 16 | 4 | 64 | 6.8703 | 429.4 | 596 |
| 16 | 8 | 128 | 13.7214 | 857.6 | 298 |
| 16 | 16 | 256 | 27.4423 | 1715.2 | 149 |

## BD3-LM

| batch | sampler | steps/block | NFE/seq | s/batch | ms/seq | tok/s |
|---|---|---|---|---|---|---|
| 1 | ancestral | 1 | 16 | 0.0939 | 93.9 | 2726 |
| 1 | ancestral | 2 | 32 | 0.2035 | 203.5 | 1258 |
| 1 | ancestral | 4 | 64 | 0.4003 | 400.3 | 640 |
| 1 | ancestral | 8 | 128 | 0.7751 | 775.1 | 330 |
| 1 | ancestral | 16 | 256 | 1.4737 | 1473.7 | 174 |
| 1 | ancestral | 32 | 512 | 2.9858 | 2985.8 | 86 |
| 4 | ancestral | 1 | 16 | 0.3199 | 80.0 | 3201 |
| 4 | ancestral | 2 | 32 | 0.6365 | 159.1 | 1609 |
| 4 | ancestral | 4 | 64 | 1.3337 | 333.4 | 768 |
| 4 | ancestral | 8 | 128 | 2.5778 | 644.4 | 397 |
| 4 | ancestral | 16 | 256 | 5.7919 | 1448.0 | 177 |
| 4 | ancestral | 32 | 512 | 11.8687 | 2967.2 | 86 |
| 16 | ancestral | 1 | 16 | 1.3911 | 86.9 | 2944 |
| 16 | ancestral | 2 | 32 | 2.7767 | 173.6 | 1475 |
| 16 | ancestral | 4 | 64 | 5.5562 | 347.3 | 737 |
| 16 | ancestral | 8 | 128 | 11.1092 | 694.3 | 369 |
| 16 | ancestral | 16 | 256 | 20.8005 | 1300.0 | 197 |
| 16 | ancestral | 32 | 512 | 42.1293 | 2633.1 | 97 |

## Iso-latency: which BD3-LM `steps_per_block` matches CFM

### batch = 1

Fit: `T(S) = 0.0926*S + 0.0165` s/batch (alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).

| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |
|---|---|---|---|---|---|
| 1 | 108.2 | **1.0** | 1 | 93.9 | 1.15x faster at S=1 |
| 2 | 216.3 | **2.2** | 2 | 203.5 | 1.06x faster at S=2 |
| 4 | 435.6 | **4.5** | 4 | 400.3 | 1.09x faster at S=4 |
| 8 | 876.2 | **9.3** | 8 | 775.1 | 1.13x faster at S=8 |
| 16 | 1778.4 | **19.0** | 16 | 1473.7 | 1.21x faster at S=16 |

### batch = 4

Fit: `T(S) = 0.3745*S + -0.1776` s/batch (alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).

| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |
|---|---|---|---|---|---|
| 1 | 105.8 | **1.6** | 2 | 159.1 | 1.50x slower at S=2 |
| 2 | 211.0 | **2.7** | 2 | 159.1 | 1.33x faster at S=2 |
| 4 | 421.6 | **5.0** | 4 | 333.4 | 1.26x faster at S=4 |
| 8 | 835.8 | **9.4** | 8 | 644.4 | 1.30x faster at S=8 |
| 16 | 1671.6 | **18.3** | 16 | 1448.0 | 1.15x faster at S=16 |

### batch = 16

Fit: `T(S) = 1.3070*S + 0.2370` s/batch (alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).

| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |
|---|---|---|---|---|---|
| 1 | 107.0 | **1.1** | 1 | 86.9 | 1.23x faster at S=1 |
| 2 | 214.2 | **2.4** | 2 | 173.6 | 1.23x faster at S=2 |
| 4 | 429.4 | **5.1** | 4 | 347.3 | 1.24x faster at S=4 |
| 8 | 857.6 | **10.3** | 8 | 694.3 | 1.24x faster at S=8 |
| 16 | 1715.2 | **20.8** | 16 | 1300.0 | 1.32x faster at S=16 |

