# MDLM vs CFM vs BD3-LM — optimized generation speed

- GPU: **NVIDIA GeForce RTX 5070 Laptop GPU**; PyTorch 2.7.1+cu128; sequence length 256.
- Every point is the median of 5 synchronized runs after 2 warmups.
- Matched denoiser budget: CFM/BD3-LM `S` steps per each of 16 blocks versus MDLM `16×S` full-sequence diffusion steps.
- CFM: sparse active-block vocab operations + finalized-prefix KV cache. BD3-LM: KV cache + lean generator + compiled dynamic backbone. MDLM: active-position vocab/sampling + lean generator + compiled static backbone.
- Precision: CFM=fp32, BD3-LM=bf16, MDLM=bf16.
- MDLM active-position sampling is distribution-equivalent but consumes RNG differently; BF16 may also change individual samples.

## Latency per generated sequence

| batch | matched S | total scheduled NFE | CFM ms | BD3-LM ms | MDLM ms | winner |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | 1 | 16 | 55.70 | 58.46 | 299.38 | **CFM** (1.05×) |
| 1 | 2 | 32 | 76.28 | 99.02 | 631.40 | **CFM** (1.30×) |
| 1 | 4 | 64 | 127.30 | 170.42 | 1224.72 | **CFM** (1.34×) |
| 1 | 8 | 128 | 229.47 | 317.63 | 2209.70 | **CFM** (1.38×) |
| 1 | 16 | 256 | 443.44 | 644.77 | 4293.59 | **CFM** (1.45×) |
| 4 | 1 | 16 | 13.03 | 49.23 | 302.50 | **CFM** (3.78×) |
| 4 | 2 | 32 | 19.44 | 82.55 | 587.00 | **CFM** (4.25×) |
| 4 | 4 | 64 | 31.25 | 161.48 | 1184.68 | **CFM** (5.17×) |
| 4 | 8 | 128 | 57.39 | 309.80 | 2196.34 | **CFM** (5.40×) |
| 4 | 16 | 256 | 119.53 | 527.26 | 4341.43 | **CFM** (4.41×) |

## Best throughput in the measured grid

| model | tokens/s | batch | setting | peak allocated GB |
|:---|---:|---:|:---|---:|
| CFM | **41,433.0** | 16 | 1 steps/block | 0.63 |
| BD3-LM | **7,405.8** | 16 | 1 steps/block | 1.82 |
| MDLM | **855.1** | 1 | 16 total steps | 1.72 |

## Interpretation

Across the common grid, CFM wins 10/10 points, BD3-LM wins 0/10 points, MDLM wins 0/10 points.
This is a speed comparison at matched scheduled denoiser calls, not a matched-quality comparison; generation quality must be evaluated separately.
