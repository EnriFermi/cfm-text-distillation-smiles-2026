# Equal-footing comparison: no inference caching on either side

- GPU: **NVIDIA GeForce RTX 5070 Laptop GPU**, torch 2.7.1+cu128, fp32, L=256, block_size 16
- CFM: `block_causal_sample` — a full `[clean; noisy]` 512-token forward per jump.
- BD3-LM: `use_kv_cache=False`, ancestral sampler — the prefix is recomputed every step.
- Both then run exactly `16 blocks x steps` network forwards per sequence.

## ms per 256-token sequence, at equal steps-per-block

| steps/block | CFM b1 | BD3-LM b1 | ratio | CFM b4 | BD3-LM b4 | ratio | CFM b16 | BD3-LM b16 | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 108.2 | 93.9 | **1.15x** | 105.8 | 91.6 | **1.15x** | 107.0 | 86.9 | **1.23x** |
| 2 | 216.3 | 203.5 | **1.06x** | 211.0 | 182.1 | **1.16x** | 214.2 | 173.6 | **1.23x** |
| 4 | 435.6 | 400.3 | **1.09x** | 421.6 | 358.4 | **1.18x** | 429.4 | 347.3 | **1.24x** |
| 8 | 876.2 | 775.1 | **1.13x** | 835.8 | 648.4 | **1.29x** | 857.6 | 694.3 | **1.24x** |
| 16 | 1778.4 | 1473.7 | **1.21x** | 1671.6 | 1329.3 | **1.26x** | 1715.2 | 1300.0 | **1.32x** |

`ratio` > 1 means BD3-LM is that much faster at the same number of steps.

## Iso-latency, cache-free

| batch | BD3-LM cost `T(S)` | S matching CFM NFE=1 | =2 | =4 | =8 | =16 |
|---|---|---|---|---|---|---|
| 1 | 0.0934*S s/batch | 1.2 | 2.3 | 4.7 | 9.4 | 19.1 |
| 4 | 0.3289*S s/batch | 1.3 | 2.6 | 5.1 | 10.2 | 20.3 |
| 16 | 1.3179*S s/batch | 1.3 | 2.6 | 5.2 | 10.4 | 20.8 |

## What BD3-LM's KV cache is worth (same weights, same S)

| batch | steps/block | cache off ms/seq | cache on ms/seq | speedup |
|---|---|---|---|---|
| 1 | 1 | 93.9 | 111.2 | **0.84x** |
| 1 | 2 | 203.5 | 180.2 | **1.13x** |
| 1 | 4 | 400.3 | 315.9 | **1.27x** |
| 1 | 8 | 775.1 | 589.5 | **1.31x** |
| 1 | 16 | 1473.7 | 1153.7 | **1.28x** |
| 4 | 1 | 91.6 | 60.0 | **1.53x** |
| 4 | 2 | 182.1 | 105.2 | **1.73x** |
| 4 | 4 | 358.4 | 199.1 | **1.80x** |
| 4 | 8 | 648.4 | 383.5 | **1.69x** |
| 4 | 16 | 1329.3 | 752.9 | **1.77x** |
| 16 | 1 | 86.9 | 54.7 | **1.59x** |
| 16 | 2 | 173.6 | 102.3 | **1.70x** |
| 16 | 4 | 347.3 | 196.5 | **1.77x** |
| 16 | 8 | 694.3 | 386.7 | **1.80x** |
| 16 | 16 | 1300.0 | 764.8 | **1.70x** |

