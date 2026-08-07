# CFM vs BD3-LM generation latency (TinyStories, L=256)

- GPU: **NVIDIA GeForce RTX 5070 Laptop GPU**, torch 2.7.1+cu128, fp32
- CFM: `block_causal`, block_size 16, 51,957,969 params, step 100000
- BD3-LM: block_size 16, 51,119,442 params, ema weights
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
| 1 | ancestral | 1 | 16 | 0.1112 | 111.2 | 2303 |
| 1 | ancestral | 2 | 32 | 0.1802 | 180.2 | 1420 |
| 1 | ancestral | 4 | 64 | 0.3159 | 315.9 | 810 |
| 1 | ancestral | 8 | 128 | 0.5896 | 589.5 | 434 |
| 1 | ancestral | 16 | 256 | 1.1537 | 1153.7 | 222 |
| 1 | ancestral | 32 | 512 | 2.2680 | 2268.0 | 113 |
| 1 | first_hitting | 16 | 255 | 1.1936 | 1193.6 | 214 |
| 1 | first_hitting | 32 | 255 | 1.1892 | 1189.2 | 215 |
| 4 | ancestral | 1 | 16 | 0.2398 | 60.0 | 4270 |
| 4 | ancestral | 2 | 32 | 0.4208 | 105.2 | 2433 |
| 4 | ancestral | 4 | 64 | 0.7964 | 199.1 | 1286 |
| 4 | ancestral | 8 | 128 | 1.5342 | 383.5 | 668 |
| 4 | ancestral | 16 | 256 | 3.0118 | 752.9 | 340 |
| 4 | ancestral | 32 | 512 | 5.9905 | 1497.6 | 171 |
| 4 | first_hitting | 16 | 255 | 3.0565 | 764.1 | 335 |
| 4 | first_hitting | 32 | 255 | 3.0390 | 759.8 | 337 |
| 16 | ancestral | 1 | 16 | 0.8755 | 54.7 | 4678 |
| 16 | ancestral | 2 | 32 | 1.6369 | 102.3 | 2502 |
| 16 | ancestral | 4 | 64 | 3.1446 | 196.5 | 1302 |
| 16 | ancestral | 8 | 128 | 6.1877 | 386.7 | 662 |
| 16 | ancestral | 16 | 256 | 12.2367 | 764.8 | 335 |
| 16 | ancestral | 32 | 512 | 24.4175 | 1526.1 | 168 |
| 16 | first_hitting | 16 | 255 | 12.2608 | 766.3 | 334 |
| 16 | first_hitting | 32 | 255 | 11.7469 | 734.2 | 349 |
| 32 | ancestral | 1 | 16 | 1.7157 | 53.6 | 4775 |
| 32 | ancestral | 2 | 32 | 3.2143 | 100.5 | 2549 |
| 32 | ancestral | 4 | 64 | 6.2235 | 194.5 | 1316 |
| 32 | ancestral | 8 | 128 | 10.3249 | 322.6 | 793 |
| 32 | ancestral | 16 | 256 | 20.4702 | 639.7 | 400 |
| 32 | ancestral | 32 | 512 | 43.9621 | 1373.8 | 186 |
| 32 | first_hitting | 16 | 255 | 21.5647 | 673.9 | 380 |
| 32 | first_hitting | 32 | 255 | 22.1957 | 693.6 | 369 |

## Iso-latency: which BD3-LM `steps_per_block` matches CFM

### batch = 1

Fit: `T(S) = 0.0696*S + 0.0385` s/batch (alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).

| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |
|---|---|---|---|---|---|
| 1 | 108.2 | **1.0** | 1 | 111.2 | 1.03x slower at S=1 |
| 2 | 216.3 | **2.6** | 2 | 180.2 | 1.20x faster at S=2 |
| 4 | 435.6 | **5.7** | 4 | 315.9 | 1.38x faster at S=4 |
| 8 | 876.2 | **12.0** | 16 | 1153.7 | 1.32x slower at S=16 |
| 16 | 1778.4 | **25.0** | 32 | 2268.0 | 1.28x slower at S=32 |

### batch = 4

Fit: `T(S) = 0.1855*S + 0.0513` s/batch (alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).

| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |
|---|---|---|---|---|---|
| 1 | 105.8 | **2.0** | 2 | 105.2 | 1.01x faster at S=2 |
| 2 | 211.0 | **4.3** | 4 | 199.1 | 1.06x faster at S=4 |
| 4 | 421.6 | **8.8** | 8 | 383.5 | 1.10x faster at S=8 |
| 8 | 835.8 | **17.7** | 16 | 752.9 | 1.11x faster at S=16 |
| 16 | 1671.6 | **35.8** | 32 | 1497.6 | 1.12x faster at S=32 |

### batch = 16

Fit: `T(S) = 0.7593*S + 0.1109` s/batch (alpha = per-block-step cost x 16 blocks, beta = the 16 KV-cache appends).

| CFM NFE | CFM ms/seq | equal-speed S | nearest measured S | BD3-LM ms/seq | BD3-LM is |
|---|---|---|---|---|---|
| 1 | 107.0 | **2.1** | 2 | 102.3 | 1.05x faster at S=2 |
| 2 | 214.2 | **4.4** | 4 | 196.5 | 1.09x faster at S=4 |
| 4 | 429.4 | **8.9** | 8 | 386.7 | 1.11x faster at S=8 |
| 8 | 857.6 | **17.9** | 16 | 764.8 | 1.12x faster at S=16 |
| 16 | 1715.2 | **36.0** | 32 | 1526.1 | 1.12x faster at S=32 |

