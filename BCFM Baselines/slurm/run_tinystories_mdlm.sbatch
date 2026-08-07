#!/bin/bash
#SBATCH --job-name=bcfm-ts-mdlm
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# Cluster-specific: uncomment / adjust for your site, or export SBATCH_PARTITION / SBATCH_ACCOUNT.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --requeue

# Single-run job: the MDLM baseline on TinyStories, nothing else.
#
# Companion to run_tinystories_ar.sbatch and run_tinystories_bd3lm.sbatch. It
# touches exactly one run directory (runs/mdlm_tinystories_seed<SEED>), so the
# three TinyStories models can be submitted independently.
#
# This is also the prerequisite for the BD3-LM job: BD3-LM initializes its
# backbone from this run's checkpoints/last.pt.
#
# TinyStories is GPT-2 BPE tokenized (50,257 ids; mask/model-only AR BOS 50257;
# tokenizer BOS/EOS <|endoftext|> is 50256). Preparation needs `tiktoken` plus one download of the
# BPE files into TIKTOKEN_CACHE_DIR; training, evaluation, and detokenization
# then read the id -> bytes table out of data/tinystories/meta.pkl and need
# neither.
#
# The GPT-2 vocabulary makes the logits (B x 256 x 50258) the largest activation
# in the step (~0.12GB per sequence in bf16 at the 384-wide backbone), so
# configs/tinystories/mdlm.yaml uses a 32x4 micro-batch/accumulation split for a
# global batch of 128 -- the CFM reference's batch, and therefore the same
# 3.28e9-token budget over 100k steps.
#
# Validation additionally measures Gen-PPL (GPT-2 Large) and MAUVE on 256
# samples, generated at 64 denoising steps so the cost stays bounded; the final
# evaluation re-measures both on 512 samples at the full 5000-step budget.
#
# Tunables (via --export=ALL,KEY=VALUE):
#   SEED                     run seed and name suffix (default 12345)
#   TINYSTORIES_DATA_DIR     prepared corpus (default data/tinystories)
#   TINYSTORIES_SOURCE_TRAIN / TINYSTORIES_SOURCE_VALID
#                            local TinyStories-*.txt, skipping the ~1.9GB download
#   TINYSTORIES_MAX_TRAIN_BYTES  cap the training text for a smaller run
#   BATCH_SIZE / GRAD_ACCUM  micro-batch and accumulation (keep the product at 128)
#   MAX_STEPS / VAL_EVERY    training length and validation cadence
#   NUM_SAMPLES / EVAL_BATCHES   evaluation sample count and test batches
#   EVAL_CHECKPOINT_NAME     last.pt (default) or best.pt
#   SKIP_COMPLETED=0         re-run even when evaluation.json already exists
#   RUN_NAME                 override the run directory name

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Keep the GPT-2 BPE and the Gen-PPL / MAUVE judge weights inside the project so
# compute nodes never need $HOME or the network after the first job.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$TIKTOKEN_CACHE_DIR" "$HF_HOME"

SEED="${SEED:-12345}"
TINYSTORIES_DATA_DIR="${TINYSTORIES_DATA_DIR:-data/tinystories}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_NAME="${RUN_NAME:-mdlm_tinystories_seed${SEED}}"
CONFIG="${CONFIG:-configs/tinystories/mdlm.yaml}"

# ---------------------------------------------------------------------------
# Python environment (self-building venv + pinned requirements).
# The project requires Python >=3.10,<3.14. Many clusters expose only `python3`
# (often 3.9/3.8) or need `module load`, so probe candidates AND verify version.
# ---------------------------------------------------------------------------
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.venv}"

python_is_compatible() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' \
    >/dev/null 2>&1
}

if [[ -n "${PYTHON_BOOTSTRAP:-}" ]]; then
  PYTHON_CANDIDATES=("$PYTHON_BOOTSTRAP")
else
  PYTHON_CANDIDATES=(python3.11 python3.12 python3.10 python3.13 python3 python)
fi
BOOTSTRAP=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1 && python_is_compatible "$candidate"; then
    BOOTSTRAP="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$BOOTSTRAP" ]]; then
  echo "ERROR: Python 3.10--3.13 was not found." >&2
  echo "Load it before sbatch (for example: module load python/3.11)," >&2
  echo "or submit with --export=ALL,PYTHON_BOOTSTRAP=/path/to/python3.11" >&2
  exit 2
fi
echo "Bootstrap:  $BOOTSTRAP ($("$BOOTSTRAP" --version 2>&1))"

if [[ -x "$ENV_DIR/bin/python" ]] && ! python_is_compatible "$ENV_DIR/bin/python"; then
  OLD_ENV="${ENV_DIR}.incompatible-${SLURM_JOB_ID:-local}"
  echo "Moving incompatible environment to $OLD_ENV"
  mv "$ENV_DIR" "$OLD_ENV"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Creating Python environment at $ENV_DIR"
  "$BOOTSTRAP" -m venv "$ENV_DIR" || "$BOOTSTRAP" -m venv --without-pip "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "pip missing in venv; bootstrapping it"
  if ! "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$ENV_DIR/get-pip.py" \
      || wget -qO "$ENV_DIR/get-pip.py" https://bootstrap.pypa.io/get-pip.py
    "$PYTHON" "$ENV_DIR/get-pip.py"
  fi
fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements-a100.txt"

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

# ===========================================================================
# 1. DATA  (re-encodes automatically if the bins predate the GPT-2 tokenizer)
# ===========================================================================
echo "== preparing TinyStories (gpt2) -> $TINYSTORIES_DATA_DIR =="
prep_args=(--output-dir "$TINYSTORIES_DATA_DIR")
[[ -n "${TINYSTORIES_SOURCE_TRAIN:-}" ]] && prep_args+=(--source-train "$TINYSTORIES_SOURCE_TRAIN")
[[ -n "${TINYSTORIES_SOURCE_VALID:-}" ]] && prep_args+=(--source-valid "$TINYSTORIES_SOURCE_VALID")
[[ -n "${TINYSTORIES_MAX_TRAIN_BYTES:-}" ]] && prep_args+=(--max-train-bytes "$TINYSTORIES_MAX_TRAIN_BYTES")
srun --unbuffered "$PYTHON" -m bcfm_baselines.data.prepare_tinystories "${prep_args[@]}"

# ===========================================================================
# 2. TRAIN + EVALUATE  (MDLM only)
# ===========================================================================
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/${EVAL_CHECKPOINT_NAME:-last.pt}"
if [[ "$SKIP_COMPLETED" == "1" && -f "$RUN_DIR/evaluation.json" ]]; then
  echo "== $RUN_NAME is already evaluated; nothing to do =="
  exit 0
fi
mkdir -p "$RUN_DIR"

overrides=()
[[ -n "${BATCH_SIZE:-}" ]] && overrides+=(--set "training.batch_size=$BATCH_SIZE")
[[ -n "${GRAD_ACCUM:-}" ]] && overrides+=(--set "training.gradient_accumulation_steps=$GRAD_ACCUM")
[[ -n "${MAX_STEPS:-}" ]] && overrides+=(--set "training.max_steps=$MAX_STEPS")
[[ -n "${VAL_EVERY:-}" ]] && overrides+=(--set "training.validation_every=$VAL_EVERY")
overrides+=(--set "training.seed=$SEED")
overrides+=(--set "dataset.data_dir=$TINYSTORIES_DATA_DIR")

echo "== training $RUN_NAME ($CONFIG) =="
srun --unbuffered "$PYTHON" -m bcfm_baselines.train \
  --config "$CONFIG" --run-dir "$RUN_DIR" --resume auto \
  ${overrides[@]+"${overrides[@]}"}

echo "== evaluating $RUN_NAME =="
srun --unbuffered "$PYTHON" -m bcfm_baselines.evaluate \
  --checkpoint "$CHECKPOINT" --output "$RUN_DIR/evaluation.json" \
  --eval-batch-size "$EVAL_BATCH_SIZE" --eval-batches "$EVAL_BATCHES" \
  --num-samples "$NUM_SAMPLES"

echo "== done: $RUN_DIR/evaluation.json =="
