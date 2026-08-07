# BCFM text-generation baselines

Standalone, directly comparable Text8 implementations of:

- decoder-only autoregressive Transformer (AR);
- Masked Diffusion Language Model (MDLM);
- Block Discrete Denoising Diffusion Language Model (BD3-LM).

The three methods share one rotary Transformer and one data/evaluation pipeline
per dataset. The neighboring Semicat/CFM repository is not copied or retrained
here; its existing checkpoints are a reference result.

**The training protocol is aligned to that CFM reference**, so the numbers are
comparable rather than merely adjacent. Per dataset the backbone, global batch,
token budget, optimizer, and LR schedule match it: Text8 uses 768/12/12/3072
(92.2M params) at global batch 2048 for 500k steps; TinyStories uses
384/6/6/1536 (51.1M params) at global batch 128 for 100k steps. AdamW betas are
the torch defaults `(0.9, 0.999)`, the LR is warmed up linearly for 500
optimizer steps from `1e-3 x lr` and then held **constant** (`training.lr_decay:
cosine` restores a decaying tail). AR keeps CFM's no-EMA protocol; MDLM and
BD3-LM deliberately use the paper's EMA `0.9999`, including EMA initialization
and validation. See `docs/cfm_parity.md` for this and every other controlled
difference.

AR uses model-only ID 27 as a BOS context token, predicts all 256 dataset
characters, and never emits that ID. This is an explicit Text8 window-boundary
adaptation of the standard shifted causal objective, not an extra data token.
MDLM and BD3-LM use the same ID as their absorbing mask state. Their research
configs maintain EMA `0.9999`; schedule search, validation, sampling, and final
evaluation use EMA while online weights remain available for exact resume.
Likelihood evaluation is seeded and token-weighted, so repeating an evaluation
with the same checkpoint and seed reproduces the same AR NLL or diffusion NELBO
estimate.

Both datasets share the same three models, trainer, and evaluator. Text8 is
character-tokenized (vocabulary 27, mask id 27); TinyStories uses the frozen
GPT-2 byte-level BPE (vocabulary 50257, mask/model-only AR BOS id 50257,
tokenizer BOS/EOS `<|endoftext|>` id 50256
between stories), selected by `dataset.name` in the config. TinyStories is
**packed**, not windowed: stories are concatenated with `<|endoftext|>` after
each, cut into 254-token pieces, and each piece is wrapped as
`[<|endoftext|>, 254 tokens, <|endoftext|>]` to fill one 256-token block, using
the same sequence layout as CFM. Blocks are non-overlapping and the corpus reads them
block-aligned, so `dataset.sequence_length` must equal the `--block-length` used
at preparation time. Preparing
TinyStories needs `tiktoken` and one download of the BPE files; the id -> bytes
table is then frozen into `data/tinystories/meta.pkl`, so training, evaluation,
and detokenization need neither.

Because TinyStories is BPE-tokenized, its reported NLL is **per token, not per
character** — the `bits_per_character` key is bits per token there. The
`split_character_counts` / `split_token_counts` recorded in
`dataset_metadata.json` convert between the two after the fact. The tokenizer
itself is a frozen merge table with no trainable parameters, but the vocabulary
it implies sizes two layers that do: the 50258-way embedding and output head are
large enough that TinyStories uses a narrower backbone (384/6/6) than Text8
(768/12/12) to land at a comparable 51.1M vs 92.2M parameters. Compare within a
dataset, not across.

## Generative evaluation: Gen-PPL and MAUVE

Every validation and every offline evaluation reports, alongside the likelihood:

- **`gen_ppl`** — generative perplexity of the model's own samples under
  **GPT-2 Large**. `bcfm_baselines/metrics/text_dist.py` is a port of the CFM
  reference's `TextMetrics.compute_mean_gen_ppl`, down to the tokenizer,
  padding/truncation, and the "score every content token plus the first EOS"
  mask; the two implementations return bit-identical values on the same texts.
- **`mauve`** (and `mauve_frontier_integral`) — MAUVE between the generated
  samples and human text drawn from the validation split, with GPT-2 Large
  features and scaling factor 5.
- **`gen_entropy`** — the reference's per-batch unigram entropy over token IDs.

Configured under `generative_evaluation` in each dataset's `base.yaml`. During
training it draws 256 samples at a reduced budget: MDLM uses 64 steps and BD3
uses 64 total ancestral NFEs (`1/2/4/8` steps per block for B=`4/8/16/32`). AR
still requires 256 cached causal calls; CFM's own one-step sampler is not changed.
The offline `evaluate` run re-measures on `--num-samples` (default 512) with
BD3's paper first-hitting sampler and float64 corrected categorical draw. MAUVE is
noisy at a few hundred samples — treat the training-time curve as a trend and
the final number as the result. Set `generative_evaluation.enabled: false` to
skip both judges.

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

### Single run: AR on TinyStories

`slurm/run_tinystories_ar.sbatch` is the same machinery narrowed to one run — it
prepares TinyStories (re-encoding it if the bins predate the GPT-2 tokenizer),
trains `configs/tinystories/ar.yaml`, and evaluates it. It touches only
`runs/<YYYYmmdd_HHMMSS>_ar_tinystories_seed<SEED>` and never trains MDLM or
BD3-LM, so it can be submitted on its own or next to the multi-model jobs. The
A100 default is `BATCH_SIZE=128,GRAD_ACCUM=1`, preserving the CFM-matched global
batch of 128 while avoiding four small accumulation passes.

AR uses the same TinyStories core Transformer capacity as CFM: width 384, six
blocks, six heads, MLP width 1536, conditioning width 128, dropout 0.1, rotary
positions, sequence length 256, and a zero-initialized output head. AdamW,
`1e-4` LR, `0.1` weight decay, 500-step linear warmup, bf16, clipping 1.0,
100K updates, and the 3.28B-token budget match too. Necessary baseline-specific
differences are explicit: AR uses a causal mask, a discrete token embedding, and
one model-only context BOS row (50,258 backbone IDs versus 50,257 data/output
IDs). Validation cadence is the requested 5K rather than CFM's original 1K; it
does not alter optimizer or scheduler steps.

```bash
sbatch slurm/run_tinystories_ar.sbatch
```

Overrides on top of the shared ones: `EVAL_BATCH_SIZE`, `MAX_STEPS`, `VAL_EVERY`,
`BATCH_SIZE` / `GRAD_ACCUM` (keep the product at the dataset's global batch),
`SKIP_COMPLETED=0`, `RUN_TIMESTAMP`, `RUN_NAME`, `CONFIG`. To resume, reuse the
`RUN_NAME` printed in the original job log.

### Single run: MDLM and BD3-LM on TinyStories

`slurm/run_tinystories_mdlm.sbatch` and `slurm/run_tinystories_bd3lm.sbatch` are
the corresponding jobs for the other two models. Paper-style BD3-LM must be
initialized from a completed, same-config MDLM EMA checkpoint. The BD3 script
checks the dataset, tokenizer, complete backbone shape/config, EMA weights, and
training step before doing any expensive work. By default it automatically
trains or resumes the TinyStories MDLM prerequisite when it is missing or
partial. An incompatible checkpoint (for example Text8 weights in the default
path) is preserved; the job builds and reuses
`runs/mdlm_tinystories_gpt2_h384_l6_seed<SEED>` instead.

```bash
sbatch slurm/run_tinystories_bd3lm.sbatch                        # MDLM first if needed, then block size 16
sbatch --export=ALL,BLOCK_SIZE=4 slurm/run_tinystories_bd3lm.sbatch
```

`BLOCK_SIZE` selects `configs/tinystories/bd3lm_b<N>.yaml` (4, 8, 16, or 32);
`BD3_PRETRAIN_CHECKPOINT` points the initialization somewhere else, while
`AUTO_TRAIN_MDLM=0` requests fail-fast behavior instead of automatic prerequisite
training. `slurm/run_tinystories_mdlm.sbatch` remains available when MDLM should
be scheduled separately. A resumed BD3-LM run reads its own checkpoint and does
not need the MDLM again. The
BD3-only job creates a timestamped directory by default, for example
`runs/20260731_173000_bd3lm_b16_tinystories_seed12345`. To resume that exact run
after submitting a new job, reuse the name printed in its log:

```bash
sbatch --export=ALL,RUN_NAME=20260731_173000_bd3lm_b16_tinystories_seed12345 \
  slurm/run_tinystories_bd3lm.sbatch
```

Alternatively, export the original `RUN_TIMESTAMP`; setting either variable
prevents a new timestamp from selecting a fresh run directory.

The timestamped directory also contains `job_<SLURM_JOB_ID>.log`, so the job can
be followed with `tail -f runs/<RUN_NAME>/job_*.log` without hunting for the
submit-directory Slurm log.

Research BD3-LM configs deliberately require an MDLM checkpoint: published
BD3-LM training initializes from one. The loader copies the compatible backbone
from this repository's same-dataset MDLM checkpoint; the official OpenWebText
checkpoint is not tokenizer/architecture compatible.

TinyStories runs write lightweight training/checkpoint state every 10/1,000
optimizer steps respectively, while the full validation suite (likelihood,
samples, Gen-PPL, and MAUVE) runs every 5,000 steps. BD3 adaptive schedule search
starts at step 5,000 and repeats at each of those validation points.

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

`training.jsonl` stores the durable train curve (NELBO, estimated PPL, rolling
100-step NELBO variance, LR, gradient norm, throughput, and selected mask
bounds). `validation.jsonl` stores full-interval NELBO/PPL, every candidate's
schedule variance, Gen-PPL, MAUVE, entropy, sampler and NFE. Human-readable
samples for each validation live in `validation_samples/step_<N>.json` and are
also added to TensorBoard. Checkpoints are flushed before the expensive judge
models run, so a judge failure does not lose the preceding training interval.

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
