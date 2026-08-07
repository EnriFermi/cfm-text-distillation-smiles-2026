# CFM vs BD3-LM — FP32 quality-neutral optimizations

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- PyTorch: 2.7.1+cu128
- Both inference paths: FP32, autocast off, TF32 off
- Sequence length 256, block size 16
- Median of 5 synchronized runs after 2 warmups
- Checkpoints: `2 models/cfm_mod/last.ckpt` and `2 models/bd3lm/last.pt`

## Included optimizations

CFM:

- clean one-hot projection replaced by the equivalent embedding lookup;
- noisy projection restricted to the active block;
- vocabulary head restricted to the active block;
- finalized clean-prefix KV cache;
- inference mode; no whole-body `torch.compile` on the dynamic cached path.

BD3-LM:

- finalized-prefix KV cache;
- FP32 `torch.compile(dynamic=True)` on the backbone;
- eval-only lean generator that omits optional entropy/mask/time diagnostics.
  Its sampled tokens matched the original generator in all 16 checkpoint checks.

Excluded: BF16/FP16, TF32, changed NFE, changed sampling strategy, changed top-p,
and any change to model weights. Cache/compiler paths are mathematically the same
samplers; different FP32 reduction order may change individual tokens.

## Equal steps per block — ancestral BD3-LM

`speedup` is BD3 latency divided by CFM latency, so values above 1 favor CFM.

| batch | steps/block | CFM ms/seq | BD3-LM ms/seq | CFM speedup |
|---:|---:|---:|---:|---:|
| 1 | 1 | 51.34 | 63.82 | 1.24x |
| 1 | 2 | 75.05 | 104.61 | 1.39x |
| 1 | 4 | 127.33 | 189.21 | 1.49x |
| 1 | 8 | 231.10 | 355.98 | 1.54x |
| 1 | 16 | 440.37 | 684.59 | 1.55x |
| 4 | 1 | 13.16 | 52.77 | 4.01x |
| 4 | 2 | 19.24 | 93.40 | 4.86x |
| 4 | 4 | 33.35 | 174.22 | 5.22x |
| 4 | 8 | 63.40 | 336.03 | 5.30x |
| 4 | 16 | 120.94 | 666.57 | 5.51x |
| 16 | 1 | 6.83 | 49.40 | 7.23x |
| 16 | 2 | 11.79 | 92.02 | 7.81x |
| 16 | 4 | 21.84 | 176.98 | 8.10x |
| 16 | 8 | 41.76 | 347.57 | 8.32x |
| 16 | 16 | 81.65 | 688.27 | 8.43x |
| 32 | 1 | 6.46 | 42.46 | 6.57x |
| 32 | 2 | 11.20 | 90.64 | 8.09x |
| 32 | 4 | 20.67 | 174.61 | 8.45x |
| 32 | 8 | 39.74 | 343.21 | 8.64x |
| 32 | 16 | 77.80 | 679.67 | 8.74x |

CFM is faster in all 20 measured equal-step points.

## CFM-16 vs official BD3 first-hitting-16

These settings have almost equal total NFE: CFM 256, BD3-LM 255.

| batch | CFM ms/seq | BD3 first-hitting ms/seq | CFM speedup |
|---:|---:|---:|---:|
| 1 | 440.37 | 700.82 | 1.59x |
| 4 | 120.94 | 666.16 | 5.51x |
| 16 | 81.65 | 651.02 | 7.97x |
| 32 | 77.80 | 674.84 | 8.67x |

## Throughput and memory

- Best measured throughput at one step/block: CFM 39,635.8 tok/s versus
  BD3-LM 6,028.8 tok/s, both at batch 32 (CFM 6.57x higher).
- At four steps/block and batch 32, peak allocated memory is 0.95 GB for CFM
  versus 2.86 GB for BD3-LM.

This is a latency comparison at fixed sampler settings, not a matched-quality
evaluation between the two model families. The optimizations do not alter the
mathematical sampler, but equal steps do not by themselves establish equal text quality.
