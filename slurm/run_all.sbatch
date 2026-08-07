#!/bin/bash
#SBATCH --job-name=bcfm-all
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
# Cluster-specific: uncomment / adjust for your site, or export SBATCH_PARTITION / SBATCH_ACCOUNT.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --requeue

# The ONE job for everything.  It runs the full pipeline first on Text8, then on
# TinyStories:
#
#   for each dataset:  prepare -> AR (train+eval) -> MDLM (train+eval)
#                      -> BD3-LM (train+eval, initialized from that dataset's MDLM)
#   finally:           aggregate every runs/*/evaluation.json
#
# Fully resumable: every training stage uses `--resume auto`, and a stage that
# already reached max_steps exits immediately.  If the job hits the wall clock,
# just resubmit it (`sbatch slurm/run_all.sbatch`) and it continues where it
# stopped.  Order guarantees each BD3-LM starts only after its dataset's MDLM
# checkpoint exists.
#
# Tunables (via --export=ALL,KEY=VALUE):
#   SEED, NUM_SAMPLES, EVAL_BATCHES, AGGREGATE_OUTPUT
#   BATCH_SIZE, GRAD_ACCUM, MAX_STEPS, VAL_EVERY  (per-run training overrides.
#     Global batch matches the CFM reference per dataset: 2048 on Text8 and 128
#     on TinyStories. Keep the BATCH_SIZE x GRAD_ACCUM product at that value and
#     raise/lower BATCH_SIZE against nvidia-smi.)
#   TEXT8_DATA_DIR, TINYSTORIES_DATA_DIR
#   TINYSTORIES_SOURCE_TRAIN / TINYSTORIES_SOURCE_VALID  (skip the ~1.9GB download)
#   TINYSTORIES_MAX_TRAIN_BYTES  (cap training bytes for a smaller run)
#   EVAL_CHECKPOINT_NAME  (last.pt [default] or best.pt)
#   PYTHON_BOOTSTRAP  (path to python3.10--3.13 if .venv must be built)
#
# Training writes both last.pt and best.pt per run, and TensorBoard event files
# under runs/<run>/tensorboard (view with: tensorboard --logdir runs).

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Keep the GPT-2 BPE inside the project so compute nodes never need $HOME or the
# network after the first TinyStories preparation.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
mkdir -p "$TIKTOKEN_CACHE_DIR"

SEED="${SEED:-12345}"
TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-data/text8}"
TINYSTORIES_DATA_DIR="${TINYSTORIES_DATA_DIR:-data/tinystories}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
AGGREGATE_OUTPUT="${AGGREGATE_OUTPUT:-runs/aggregate.json}"

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

# A venv left by an incompatible interpreter (e.g. a partial earlier attempt) is
# moved aside rather than reused.
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

# Some cluster Python builds create venvs without pip (stripped ensurepip);
# bootstrap it via ensurepip, falling back to get-pip.py (needs node internet).
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "pip missing in venv; bootstrapping it"
  if ! "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$ENV_DIR/get-pip.py" \
      || wget -qO "$ENV_DIR/get-pip.py" https://bootstrap.pypa.io/get-pip.py
    "$PYTHON" "$ENV_DIR/get-pip.py"
  fi
fi
"$PYTHON" -m pip install --upgrade pip

# Also run for an existing env: an interrupted first job may leave a venv with
# only part of the requirements installed.
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements-a100.txt"

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
# run_stage CONFIG RUN_NAME [ENV=VALUE ...]: train (resumable) then evaluate.
# Training writes both checkpoints/last.pt (final EMA) and checkpoints/best.pt
# (lowest validation NLL/NELBO); EVAL_CHECKPOINT_NAME selects which to evaluate.
run_stage() {
  local config="$1" run="$2"; shift 2
  local run_dir="$PROJECT_DIR/runs/$run"
  local ckpt="$run_dir/checkpoints/${EVAL_CHECKPOINT_NAME:-last.pt}"
  mkdir -p "$run_dir"
  local overrides=()
  [[ -n "${BATCH_SIZE:-}" ]] && overrides+=(--set "training.batch_size=$BATCH_SIZE")
  [[ -n "${GRAD_ACCUM:-}" ]] && overrides+=(--set "training.gradient_accumulation_steps=$GRAD_ACCUM")
  [[ -n "${MAX_STEPS:-}" ]] && overrides+=(--set "training.max_steps=$MAX_STEPS")
  [[ -n "${VAL_EVERY:-}" ]] && overrides+=(--set "training.validation_every=$VAL_EVERY")
  echo "== training $run ($config) =="
  env "$@" srun --unbuffered "$PYTHON" -m bcfm_baselines.train \
    --config "$config" --run-dir "$run_dir" --resume auto \
    ${overrides[@]+"${overrides[@]}"}
  echo "== evaluating $run =="
  srun --unbuffered "$PYTHON" -m bcfm_baselines.evaluate \
    --checkpoint "$ckpt" --output "$run_dir/evaluation.json" \
    --num-samples "$NUM_SAMPLES" --eval-batches "$EVAL_BATCHES"
}

# run_dataset LABEL AR_CFG MDLM_CFG BD3_CFG: AR, MDLM, then BD3-LM from the MDLM.
run_dataset() {
  local label="$1" ar_cfg="$2" mdlm_cfg="$3" bd3_cfg="$4"
  local mdlm_run="mdlm_${label}_seed${SEED}"
  local mdlm_ckpt="$PROJECT_DIR/runs/$mdlm_run/checkpoints/last.pt"
  run_stage "$ar_cfg"   "ar_${label}_seed${SEED}"
  run_stage "$mdlm_cfg" "$mdlm_run"
  if [[ ! -f "$mdlm_ckpt" ]]; then
    echo "MDLM checkpoint missing at $mdlm_ckpt; cannot initialize BD3-LM" >&2
    exit 3
  fi
  run_stage "$bd3_cfg" "bd3lm_b16_${label}_seed${SEED}" "BD3_PRETRAIN_CHECKPOINT=$mdlm_ckpt"
}

# ===========================================================================
# 1. TEXT8
# ===========================================================================
if [[ ! -f "$TEXT8_DATA_DIR/train.bin" ]]; then
  echo "== preparing Text8 -> $TEXT8_DATA_DIR =="
  srun --unbuffered "$PYTHON" -m bcfm_baselines.data.prepare_text8 \
    --output-dir "$TEXT8_DATA_DIR"
fi
run_dataset text8 \
  configs/text8/ar.yaml configs/text8/mdlm.yaml configs/text8/bd3lm_b16.yaml

# ===========================================================================
# 2. TINYSTORIES
# ===========================================================================
# Always invoked: prepare_tinystories returns immediately when the corpus is
# already GPT-2 tokenized, and re-encodes it when the bins are older byte-level
# ones, which a plain train.bin existence check would silently accept.
echo "== preparing TinyStories (gpt2) -> $TINYSTORIES_DATA_DIR =="
prep_args=(--output-dir "$TINYSTORIES_DATA_DIR")
[[ -n "${TINYSTORIES_SOURCE_TRAIN:-}" ]] && prep_args+=(--source-train "$TINYSTORIES_SOURCE_TRAIN")
[[ -n "${TINYSTORIES_SOURCE_VALID:-}" ]] && prep_args+=(--source-valid "$TINYSTORIES_SOURCE_VALID")
[[ -n "${TINYSTORIES_MAX_TRAIN_BYTES:-}" ]] && prep_args+=(--max-train-bytes "$TINYSTORIES_MAX_TRAIN_BYTES")
srun --unbuffered "$PYTHON" -m bcfm_baselines.data.prepare_tinystories "${prep_args[@]}"
run_dataset tinystories \
  configs/tinystories/ar.yaml configs/tinystories/mdlm.yaml configs/tinystories/bd3lm_b16.yaml

# ===========================================================================
# 3. AGGREGATE
# ===========================================================================
echo "== aggregating =="
mapfile -t INPUTS < <(find runs -name evaluation.json -type f | sort)
if [[ ${#INPUTS[@]} -eq 0 ]]; then echo "No evaluation files found" >&2; exit 2; fi
srun --unbuffered "$PYTHON" -m bcfm_baselines.aggregate \
  "${INPUTS[@]}" --output "$AGGREGATE_OUTPUT"

echo "== done: $AGGREGATE_OUTPUT =="
