# Baselines implementation plan

## Scope and stage gate

This standalone repository implements three baselines: AR, MDLM, and BD3-LM.
Per the researcher's clarification, the neighboring CFM/Semicat model is neither
copied nor retrained. Existing reference checkpoints/results remain outside this
directory. Stage 1 is Text8; Stage 2 (TinyStories) is now implemented as a
byte-level data layer that reuses the same models, trainer, and evaluator.

## Inspected reference architecture

Before implementation, the neighboring research repository was inspected in
full. Its Text8 comparison backbone is a 12-layer, hidden-size 768, 12-head,
4x-MLP rotary Transformer with LayerNorm, GELU, dropout 0.1, and a 128-wide time
condition. Text8 uses character IDs from `meta.pkl`, `train.bin`, and `val.bin`,
sequence length 256, overlapping windows, and split lengths
`[351563, 20000, 19063]`. Training used AdamW, BF16, gradient accumulation,
Lightning checkpoints, and Hydra configs. The repository contained no BCFM
block implementation and no AR/MDLM/BD3-LM baseline suite.

## Current architecture

- `bcfm_baselines/net/text_transformer.py`: shared Transformer with rotary
  positions, AdaLN time conditioning, causal/full/block masks, and KV caches.
- `bcfm_baselines/models/generative_text.py`: common `compute_loss`, `generate`,
  and `num_parameters` interface plus generation configuration/diagnostics.
- `bcfm_baselines/models/ar.py`: full-sequence causal cross-entropy from an
  internal BOS token and cached decoding.
- `bcfm_baselines/models/mdlm.py`: absorbing corruption, continuous-time SUBS
  NELBO, official epsilon-grid DDPM-cache sampling, and noise removal.
- `bcfm_baselines/models/bd3lm.py`: efficient concatenated clean/noisy training,
  correctness-critical block mask, official first-hitting plus ancestral
  block-AR sampling, and finalized-prefix cache.
- `bcfm_baselines/data/`: deterministic Text8 preparation, metadata/checksums,
  memory mapping, exact reference split mode, and reproducible random windows.
- `bcfm_baselines/train.py`: unified config-selected trainer with AdamW, warmup +
  cosine decay, AMP, accumulation, EMA, token-weighted validation, BD3-LM
  clipped-schedule search, MDLM-to-BD3 backbone initialization, atomic `last.pt`
  and `best.pt` checkpoints (best by validation NLL/NELBO), TensorBoard scalar
  logging, exact batch RNG resume, and run metadata.
- `sample.py`, `evaluate.py`, and `aggregate.py`: checkpoint sampling, offline
  Text8 likelihood/sample metrics, diagnostics, and multi-run summaries.
- `bcfm_baselines/data/prepare_tinystories.py` + `tinystories.py`: Stage 2
  byte-level TinyStories preparation and corpus; `data.create_corpus` selects
  the corpus from `dataset.name`.
- `configs/text8/` and `configs/tinystories/`: research-scale AR, MDLM, and
  BD3-LM block-size configs per dataset.
- `configs/smoke/`: small CPU configs for every model on both datasets.
- `slurm/run_all.sbatch`: one resumable job that prepares, trains, evaluates,
  and aggregates all models on Text8 first and TinyStories second.
- `tests/`: objective, mask, cache-equivalence, deterministic generation,
  one-step transition, checkpoint/resume, and end-to-end smoke tests.

## Shared model interface and fairness

All models implement:

```python
def compute_loss(batch, **kwargs) -> dict[str, Tensor]: ...
@torch.no_grad()
def generate(generation_config, **kwargs) -> GenerationResult: ...
def num_parameters() -> dict[str, int]: ...
```

`model.type: ar | mdlm | bd3lm` selects the objective without selecting another
training program. All research configs fix layers=12, hidden=768, heads=12,
intermediate=3072, dropout=0.1, rotary positions, context=256, initialization,
optimizer, schedule, precision, token budget, global batch, seed, and evaluation
count. Dataset token IDs remain 0..26. Diffusion uses internal mask ID 27; AR
uses the same model-only ID as BOS, scores all sequence tokens, and excludes it
from the output distribution. Attention semantics and diffusion conditioning
are the documented objective-required differences.

The resolved research backbone has exactly 92,244,764 trainable parameters for
each method. This is below the approximate 100M--110M target because Text8 has
only 28 model embeddings; retaining the inspected 12x768x12, 128-wide condition
backbone was prioritized over changing width solely to inflate parameter count.

## Text8 implementation order

1. Prepare canonical Text8 and record source/preprocessing/file hashes.
2. Validate shared backbone, interface, losses, masks, and deterministic samplers.
3. Run one-step CPU smoke pipelines including checkpoint save/load/resume.
4. Submit the Slurm prepare -> train -> evaluate chain for every model/config.
5. Re-submit one interrupted job to validate cluster-level resume.
6. Evaluate all requested step counts and aggregate equal-budget seeds.
7. Freeze Stage 1 results and only then generalize the data/token layer to
   TinyStories.

## Stage 2 (TinyStories, implemented)

The same models/trainer/evaluator are reused with one frozen tokenizer shared
across methods: a byte-level tokenizer (UTF-8 bytes, 256 symbols, mask/BOS id
256). It is frozen by construction, so no vocabulary is fit from data. The
special IDs, dataset revision (`roneneldan/TinyStories`), preprocessing version,
source hashes, and split byte counts are recorded in `dataset_metadata.json`.
Context is 256, matching Text8; 512 remains an optional future change. No model
or training code is duplicated -- only `data.create_corpus` selects the corpus
from `dataset.name`, and `configs/tinystories/*` set the byte vocabulary.

## Compatibility and research risks

- The reference repository remains external, so comparison relies on identical
  data/split/backbone properties and a common result schema rather than importing
  or changing its model code.
- The reference Text8 split is unusually small. Research configs reproduce it
  explicitly; canonical full splits require an explicit config change.
- AR exact NLL and diffusion Monte Carlo objectives are labeled separately.
  Diffusion evaluation is an epsilon-truncated NELBO/BPC estimate and is not
  presented as a strict upper bound on the full likelihood.
- Single-sample diffusion NELBO is seeded and repeatable. Scalar validation and
  test metrics are weighted by valid-token counts rather than batch counts.
- BD3-LM uses official-style clipped-schedule variance search and masked-fraction
  resampling for training, but always reports the full configured-interval
  NELBO instead of the clipped training objective.
- Research BD3-LM configs require initialization from a compatible local Text8
  MDLM checkpoint, matching the paper's pretrain-then-fine-tune workflow. Smoke
  configs remain scratch-only and are never treated as research results.
- EMA weights are saved and selected automatically for validation and inference;
  online weights remain in the checkpoint for exact optimizer resumption.
- PyTorch SDPA boolean masks are the correctness reference. Optimized kernels
  must pass mask/output-equivalence tests before use.
- AR cache is exact. BD3-LM caches only finalized time-independent clean prefix
  states and recomputes the noisy active block; first-hitting and ancestral
  tests compare it to no-cache.
- Site-specific Slurm account/partition/GPU directives may require header edits.
- Actual cluster throughput, GPU memory, preemption behavior, and long-run
  numerical stability cannot be certified by CPU smoke tests and remain part of
  the Stage 1 cluster gate.
