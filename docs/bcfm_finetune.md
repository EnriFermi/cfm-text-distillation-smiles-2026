# BCFM fine-tuning from the trained CFM

The ready-to-run entry point is:

```bash
conda activate semicat
./scripts/train_bcfm_from_baseline.sh
```

The launcher initializes the network strictly from
`baseline/s_baseline.ckpt` (source step 390000) and starts a fresh optimizer.
It does not restore the baseline optimizer/global step. A later interrupted
BCFM run should instead be continued with Lightning resume:

```bash
./scripts/train_bcfm_from_baseline.sh ckpt_path=/absolute/path/to/last.ckpt
```

The default objective is the checkpoint's upstream CFM/Lagrangian objective
(`model.sd_type=lag`; `cfm` is accepted as an alias). The block-generalized
upstream ECLD objective is available with:

```bash
./scripts/train_bcfm_from_baseline.sh model.sd_type=ecld \
  model.expected_source_sd_type=lag
```

Defaults match the checkpoint and untouched SemiCat: Text8 stride-one windows,
split `[351563, 20000, 19063]`, length 256, batch 128, Gaussian prior,
`sd_prop=0.25`, AdamW at `1e-4` with weight decay `0.01`, and no scheduler.
There is no gradient accumulation; the upstream recipe's global-norm clipping
at 1.0 is retained.
The intervention is limited to per-block times, the doubled clean/noisy
block-causal graph, and the block sampler.

Validation runs every 10,000 optimizer steps. Every such checkpoint is retained
as `checkpoints/step_XXXXXXX.ckpt`, together with `last.ckpt`; the callback is
not restricted to top-1.

On CUDA, ordinary VFM/teacher forwards use the official BD3-LM compiled
FlexAttention mask. CFM/ECLD JVP forwards cache the clean stream and use
SemiCat's custom Triton JVP attention independently over each permitted sparse
block context. This separation is required because FlexAttention itself does
not support `torch.func.jvp` in the pinned Torch version.

The Flex sparse tile is 64. On the target H100 at the actual batch-96,
12-head, doubled-length-512 shape, forward+backward measured 1.90 ms at tile 64
versus 2.16 ms at tile 128 and 4.55 ms for dense SDPA. Tiles 16/32 are not
supported by this Torch/Inductor kernel.

To re-check the untouched inheritance:

```bash
python scripts/verify_semicat_upstream_parity.py
```

The reference is SemiCat commit
`558602a0fa722514e4a6012f5c46a8ae178b3068`. All inherited tracked files must
match byte-for-byte except `.gitignore` and `environment.yaml`, whose only
purpose here is repository/dependency infrastructure.

## Verification evidence

No optimization experiment was launched and no optimizer step was performed.
The setup was verified with:

- 38 passing repository tests, including CUDA reverse-over-forward JVP,
  no-leak mask checks, and `B=L` loss equivalence for both `lag` and `ecld`;
- strict loading of all 164 checkpoint tensors / 92,491,547 parameters;
- a full batch-128 CFM forward/backward at 51.3 GiB peak allocation;
- a second in-process batch-128 forward/backward in 1.20 seconds after compile
  warm-up;
- an actual untouched Text8 CUDA dataloader batch.

Machine-readable evidence is under `artifacts/bcfm_verification/`:

- `upstream_parity.json`
- `checkpoint_initialization.json`
- `b_equals_l_checkpoint_equivalence.json`
- `attention_benchmark_batch96.json`
- `jvp_benchmark_batch32.json`
- `full_batch128_math_jvp_cuda.json`
- `full_batch128_steady.json`
- `text8_dataloader_smoke.json`
