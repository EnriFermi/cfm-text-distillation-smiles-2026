# Every optimization enabled, both models

- GPU **NVIDIA GeForce RTX 5070 Laptop GPU**, torch 2.7.1+cu128, L=256, block_size 16, 256-token sequences.
- **baseline** = each repo's untouched sampler, fp32, no caching.
- **lossless** = verified token-for-token identical to that baseline: CFM sparse-vocab + clean-KV cache (fp32, eager); BD3-LM `use_kv_cache=True` + `torch.compile`.
- **bf16** = the same, in bfloat16. A different sample, not a faster identical one.

## ms per sequence — lossless on both sides

### batch = 1

| steps/block | CFM base | CFM lossless | gain | BD3-LM base | BD3-LM lossless | gain | **winner** |
|---|---|---|---|---|---|---|---|
| 1 | 108.2 | **48.55** | 2.2x | 93.9 | **74.32** | 1.3x | **CFM 1.5x** |
| 2 | 216.3 | **72.91** | 3.0x | 203.5 | **132.43** | 1.5x | **CFM 1.8x** |
| 4 | 435.6 | **119.66** | 3.6x | 400.3 | **218.04** | 1.8x | **CFM 1.8x** |
| 8 | 876.2 | **232.36** | 3.8x | 775.1 | **419.28** | 1.8x | **CFM 1.8x** |
| 16 | 1778.4 | **433.46** | 4.1x | 1473.7 | **742.32** | 2.0x | **CFM 1.7x** |

### batch = 4

| steps/block | CFM base | CFM lossless | gain | BD3-LM base | BD3-LM lossless | gain | **winner** |
|---|---|---|---|---|---|---|---|
| 1 | 105.8 | **12.23** | 8.7x | 80.0 | **51.68** | 1.5x | **CFM 4.2x** |
| 2 | 211.0 | **18.71** | 11.3x | 159.1 | **104.95** | 1.5x | **CFM 5.6x** |
| 4 | 421.6 | **33.00** | 12.8x | 333.4 | **200.12** | 1.7x | **CFM 6.1x** |
| 8 | 835.8 | **63.85** | 13.1x | 644.4 | **385.47** | 1.7x | **CFM 6.0x** |
| 16 | 1671.6 | **120.76** | 13.8x | 1448.0 | **768.10** | 1.9x | **CFM 6.4x** |

### batch = 16

| steps/block | CFM base | CFM lossless | gain | BD3-LM base | BD3-LM lossless | gain | **winner** |
|---|---|---|---|---|---|---|---|
| 1 | 107.0 | **6.85** | 15.6x | 86.9 | **55.06** | 1.6x | **CFM 8.0x** |
| 2 | 214.2 | **11.82** | 18.1x | 173.6 | **102.79** | 1.7x | **CFM 8.7x** |
| 4 | 429.4 | **21.77** | 19.7x | 347.3 | **178.92** | 1.9x | **CFM 8.2x** |
| 8 | 857.6 | **42.36** | 20.2x | 694.3 | **390.24** | 1.8x | **CFM 9.2x** |
| 16 | 1715.2 | **83.88** | 20.4x | 1300.0 | **739.06** | 1.8x | **CFM 8.8x** |

### batch = 32

| steps/block | CFM base | CFM lossless | gain | BD3-LM base | BD3-LM lossless | gain | **winner** |
|---|---|---|---|---|---|---|---|
| 1 | OOM | **6.74** | n/a (OOM) | – | **53.91** | – | **CFM 8.0x** |
| 2 | OOM | **11.49** | n/a (OOM) | – | **100.92** | – | **CFM 8.8x** |
| 4 | OOM | **21.28** | n/a (OOM) | – | **185.58** | – | **CFM 8.7x** |
| 8 | OOM | **40.24** | n/a (OOM) | – | **354.06** | – | **CFM 8.8x** |
| 16 | OOM | **78.62** | n/a (OOM) | – | **701.07** | – | **CFM 8.9x** |

## Does bf16 buy anything on top?

| batch | steps | CFM lossless | CFM bf16 | BD3-LM lossless | BD3-LM bf16 |
|---|---|---|---|---|---|
| 1 | 1 | 48.55 | 72.28 | 74.32 | 74.97 |
| 1 | 4 | 119.66 | 166.11 | 218.04 | 206.24 |
| 1 | 16 | 433.46 | 622.27 | 742.32 | 790.63 |
| 4 | 1 | 12.23 | 17.09 | 51.68 | 53.75 |
| 4 | 4 | 33.00 | 46.04 | 200.12 | 184.65 |
| 4 | 16 | 120.76 | 171.40 | 768.10 | 712.27 |
| 16 | 1 | 6.85 | 8.13 | 55.06 | 47.07 |
| 16 | 4 | 21.77 | 26.73 | 178.92 | 163.13 |
| 16 | 16 | 83.88 | 101.94 | 739.06 | 605.37 |
| 32 | 1 | 6.74 | 6.86 | 53.91 | 42.43 |
| 32 | 4 | 21.28 | 22.93 | 185.58 | 151.70 |
| 32 | 16 | 78.62 | 87.38 | 701.07 | 582.60 |

## Peak memory, GB (steps/block = 4)

| batch | CFM base | CFM lossless | BD3-LM base | BD3-LM lossless |
|---|---|---|---|---|
| 1 | 0.59 | 0.32 | 1.16 | 1.11 |
| 4 | 1.69 | 0.38 | 1.49 | 1.32 |
| 16 | 6.09 | 0.63 | 2.81 | 1.95 |
| 32 | OOM | 0.95 | – | 2.86 |

## Throughput ceiling over the whole grid

| model | regime | best tok/s | at |
|---|---|---|---|
| CFM | baseline | **2,420** | batch 4, 1 steps |
| CFM | lossless | **37,967** | batch 32, 1 steps |
| CFM | bf16 | **37,319** | batch 32, 1 steps |
| BD3-LM | baseline | **3,201** | batch 4, 1 steps |
| BD3-LM | lossless | **4,954** | batch 4, 1 steps |
| BD3-LM | bf16 | **6,034** | batch 32, 1 steps |

