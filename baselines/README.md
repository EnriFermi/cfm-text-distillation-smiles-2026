# BCFM text-generation baselines

Vendored into the CFM repo under ``baselines/``. Eval scripts set
``BASELINES_CODE=$PWD/baselines`` by default (override with the env var).
Prepared corpora stay outside git: ``data/text8``, ``data/tinystories``
(see ``python -m bcfm_baselines.data.prepare_tinystories``).

Standalone, directly comparable Text8 implementations of:

- decoder-only autoregressive Transformer (AR);
- Masked Diffusion Language Model (MDLM);
- Block Discrete Denoising Diffusion Language Model (BD3-LM).

The three methods share one 12-layer, 768-hidden, 12-head, 3072-MLP rotary
Transformer and one Text8 data/evaluation pipeline. The neighboring Semicat/CFM
repository is not copied or retrained here; its existing checkpoints are a
reference result.

AR uses model-only ID 27 as a BOS context token, predicts all 256 dataset
characters, and never emits that ID. This is an explicit Text8 window-boundary
adaptation of the standard shifted causal objective, not an extra data token.
MDLM and BD3-LM use the same ID as their absorbing mask state. Training
maintains EMA weights (`0.9999` in research configs); sampling and evaluation
load EMA by default. Likelihood evaluation is
seeded and token-weighted, so repeating an evaluation with the same checkpoint
and seed reproduces the same AR NLL or diffusion NELBO estimate.

Both datasets share the same three models, trainer, and evaluator. Text8 is
character-tokenized (vocabulary 27, mask id 27); TinyStories uses a frozen
byte-level tokenizer (vocabulary 256, mask id 256), selected by
`dataset.name` in the config.

## One-command Slurm workflow (Text8 -> TinyStories)

The single job `slurm/run_all.sbatch` runs the entire study end to end: it
builds `.venv`, installs `requirements-a100.txt`, then for **each** dataset
prepares the data and runs AR, MDLM, and BD3-LM (each BD3-LM initialized from
that dataset's MDLM checkpoint), and finally aggregates every run. Text8 runs
first, TinyStories second.

```bash
sbatch slurm/run_all.sbatch
# add cluster directives as needed, e.g.
export SBATCH_PARTITION=gpu SBATCH_ACCOUNT=your_account
```

It is fully resumable: every training stage uses `--resume auto`, and a stage
that already reached `max_steps` exits immediately, so if the job hits the wall
clock you simply `sbatch` it again and it continues. On module-based clusters,
load Python 3.10--3.13 first or export `PYTHON_BOOTSTRAP=/path/to/python`.

Useful overrides (`sbatch --export=ALL,KEY=VALUE,... slurm/run_all.sbatch`):
`SEED`, `NUM_SAMPLES`, `EVAL_BATCHES`, `TEXT8_DATA_DIR`, `TINYSTORIES_DATA_DIR`,
`TINYSTORIES_SOURCE_TRAIN` / `TINYSTORIES_SOURCE_VALID` (skip the ~1.9 GB
download by pointing at local `TinyStories-*.txt`), and
`TINYSTORIES_MAX_TRAIN_BYTES`.

Research BD3-LM configs deliberately require an MDLM checkpoint: published
BD3-LM training initializes from one. The loader copies the compatible EMA
backbone from this repository's same-dataset MDLM checkpoint; the official
OpenWebText checkpoint is not tokenizer/architecture compatible.

### Direct commands (what the job runs)

```bash
python -m bcfm_baselines.data.prepare_text8 --output-dir data/text8
python -m bcfm_baselines.data.prepare_tinystories --output-dir data/tinystories
python -m bcfm_baselines.train --config configs/text8/ar.yaml --run-dir runs/ar_text8_seed12345
python -m bcfm_baselines.sample --checkpoint runs/ar_text8_seed12345/checkpoints/last.pt --output runs/ar_text8_seed12345/samples.json
python -m bcfm_baselines.evaluate --checkpoint runs/ar_text8_seed12345/checkpoints/last.pt --output runs/ar_text8_seed12345/evaluation.json
python -m bcfm_baselines.aggregate runs/*/evaluation.json --output runs/aggregate.json
```

## Checkpoints and logging

Every training run writes, under `runs/<run>/checkpoints/`:

- `last.pt` — the most recent state (used for `--resume auto` and the default
  for evaluation); a final `step_<N>.pt` is also saved at `max_steps`.
- `best.pt` — the checkpoint with the lowest validation metric so far (AR: exact
  NLL; MDLM/BD3-LM: epsilon-truncated NELBO estimate). The monitored metric and
  its value are recorded in the checkpoint and `run_metadata.json`.

Both are full, resumable payloads and can be passed to
`bcfm_baselines.evaluate`/`sample` via `--checkpoint`; on Slurm set
`EVAL_CHECKPOINT_NAME=best.pt` to evaluate the best instead of the last.

TensorBoard scalars (train loss/metrics, learning rate, validation metrics) are
written to `runs/<run>/tensorboard/`:

```bash
tensorboard --logdir runs
```

## CPU smoke validation

```bash
python -m bcfm_baselines.data.prepare_text8 --output-dir data/text8-smoke --smoke --force
python -m bcfm_baselines.data.prepare_tinystories --output-dir data/tinystories-smoke --smoke --force
pytest -q
```

The TinyStories smoke corpus is synthetic and needs no download; `pytest`
builds it in a temporary directory via a fixture, so the line above is only for
manual inspection.

Research training is designed for Slurm, not local interactive execution. See
`docs/text8_baselines_status.md`, `docs/mdlm.md`, and `docs/bd3lm.md` for the
objective definitions and exact operational commands.
