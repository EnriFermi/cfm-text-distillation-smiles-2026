#!/bin/bash
#SBATCH --job-name=bcfm-remaining
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
# Cluster-specific: uncomment / adjust for your site, or export SBATCH_PARTITION / SBATCH_ACCOUNT.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --requeue

# Resumption job for a partially completed slurm/run_all.sbatch.
#
# It walks the same stage list in the same order but runs only the stages that
# are not finished yet: a stage whose runs/<run>/evaluation.json exists is
# skipped outright, and a stage whose training already reached max_steps falls
# straight through to evaluation.  After the Text8 AR and MDLM runs of the
# original job, that leaves BD3-LM b16 on Text8 plus all three TinyStories runs.
#
# It also carries the fix for the crash that stopped the first job:
# bcfm_baselines/models/bd3lm.py read its mask-rate bounds back from float32
# buffers, where min_time 1e-3 becomes 1.0000000474974513e-3, so the unclipped
# interval looked clipped and the resampler demanded a masked token in every
# block -- unsatisfiable for a block whose sampled rate is epsilon:
#   RuntimeError: could not sample a block masking fraction inside the
#   configured bounds
# Sync the repository to the cluster before submitting, otherwise BD3-LM stops
# the same way again.
#
# Tunables (via --export=ALL,KEY=VALUE): identical to run_all.sbatch, plus
#   SKIP_COMPLETED=0   re-run every stage even when evaluation.json exists
#   BD3_BLOCK_SIZES    space-separated block sizes to train (default: 16)

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

SEED="${SEED:-12345}"
TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-data/text8}"
TINYSTORIES_DATA_DIR="${TINYSTORIES_DATA_DIR:-data/tinystories}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
AGGREGATE_OUTPUT="${AGGREGATE_OUTPUT:-runs/aggregate.json}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
BD3_BLOCK_SIZES="${BD3_BLOCK_SIZES:-16}"

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

# The bd3lm.py shipped with this job must be the fixed one; the buggy revision
# only fails after minutes of training, which is an expensive way to find out.
"$PYTHON" - <<'PY'
import torch
from bcfm_baselines.models.bd3lm import BlockDiffusionTextModel

backbone = dict(
    vocab_size=28, max_sequence_length=256, num_layers=1, hidden_size=32,
    num_attention_heads=4, intermediate_size=64, time_condition_size=16, dropout=0.0,
)
model = BlockDiffusionTextModel(backbone=backbone, block_size=16)
bounds = (float(model.mask_rate_min.item()), float(model.mask_rate_max.item()))
clean = torch.randint(0, 27, (128, 256))
valid = torch.ones_like(clean, dtype=torch.bool)
torch.manual_seed(0)
for _ in range(20):
    block_t = model._sample_block_times(128, 16, clean.device, bounds)
    token_t = block_t.clamp(min=model.min_time, max=1.0).repeat_interleave(16, dim=1)
    model._sample_mask(clean, valid, token_t, bounds, None)
print("BD3-LM mask sampler check: ok")
PY

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
# run_stage CONFIG RUN_NAME [ENV=VALUE ...]: train (resumable) then evaluate,
# unless the run already produced evaluation.json.
run_stage() {
  local config="$1" run="$2"; shift 2
  local run_dir="$PROJECT_DIR/runs/$run"
  local ckpt="$run_dir/checkpoints/${EVAL_CHECKPOINT_NAME:-last.pt}"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$run_dir/evaluation.json" ]]; then
    echo "== skipping $run (already evaluated) =="
    return 0
  fi
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

# run_dataset LABEL CONFIG_DIR: AR, MDLM, then one BD3-LM per requested block
# size, each initialized from that dataset's MDLM checkpoint.
run_dataset() {
  local label="$1" config_dir="$2"
  local mdlm_run="mdlm_${label}_seed${SEED}"
  local mdlm_ckpt="$PROJECT_DIR/runs/$mdlm_run/checkpoints/last.pt"
  run_stage "$config_dir/ar.yaml"   "ar_${label}_seed${SEED}"
  run_stage "$config_dir/mdlm.yaml" "$mdlm_run"
  if [[ ! -f "$mdlm_ckpt" ]]; then
    echo "MDLM checkpoint missing at $mdlm_ckpt; cannot initialize BD3-LM" >&2
    exit 3
  fi
  for block in $BD3_BLOCK_SIZES; do
    run_stage "$config_dir/bd3lm_b${block}.yaml" "bd3lm_b${block}_${label}_seed${SEED}" \
      "BD3_PRETRAIN_CHECKPOINT=$mdlm_ckpt"
  done
}

# ===========================================================================
# 1. TEXT8  (AR and MDLM are already evaluated; this reaches BD3-LM)
# ===========================================================================
if [[ ! -f "$TEXT8_DATA_DIR/train.bin" ]]; then
  echo "== preparing Text8 -> $TEXT8_DATA_DIR =="
  srun --unbuffered "$PYTHON" -m bcfm_baselines.data.prepare_text8 \
    --output-dir "$TEXT8_DATA_DIR"
fi
run_dataset text8 configs/text8

# ===========================================================================
# 2. TINYSTORIES
# ===========================================================================
if [[ ! -f "$TINYSTORIES_DATA_DIR/train.bin" ]]; then
  echo "== preparing TinyStories -> $TINYSTORIES_DATA_DIR =="
  prep_args=(--output-dir "$TINYSTORIES_DATA_DIR")
  [[ -n "${TINYSTORIES_SOURCE_TRAIN:-}" ]] && prep_args+=(--source-train "$TINYSTORIES_SOURCE_TRAIN")
  [[ -n "${TINYSTORIES_SOURCE_VALID:-}" ]] && prep_args+=(--source-valid "$TINYSTORIES_SOURCE_VALID")
  [[ -n "${TINYSTORIES_MAX_TRAIN_BYTES:-}" ]] && prep_args+=(--max-train-bytes "$TINYSTORIES_MAX_TRAIN_BYTES")
  srun --unbuffered "$PYTHON" -m bcfm_baselines.data.prepare_tinystories "${prep_args[@]}"
fi
run_dataset tinystories configs/tinystories

# ===========================================================================
# 3. AGGREGATE
# ===========================================================================
echo "== aggregating =="
mapfile -t INPUTS < <(find runs -name evaluation.json -type f | sort)
if [[ ${#INPUTS[@]} -eq 0 ]]; then echo "No evaluation files found" >&2; exit 2; fi
srun --unbuffered "$PYTHON" -m bcfm_baselines.aggregate \
  "${INPUTS[@]}" --output "$AGGREGATE_OUTPUT"

echo "== done: $AGGREGATE_OUTPUT =="
