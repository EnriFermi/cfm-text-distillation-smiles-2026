#!/bin/bash
#SBATCH --job-name=bcfm-ts-bd3lm
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# Cluster-specific: uncomment / adjust for your site, or export SBATCH_PARTITION / SBATCH_ACCOUNT.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --requeue

# BD3-LM job on TinyStories. If its required same-config MDLM checkpoint does
# not exist yet, this job trains/resumes that prerequisite first and then starts
# BD3-LM. It never substitutes a Text8 or otherwise incompatible checkpoint.
#
# The counterpart of run_tinystories_ar.sbatch. It touches exactly one run
# timestamped directory
# (runs/<YYYYmmdd_HHMMSS>_bd3lm_b<BLOCK_SIZE>_tinystories_seed<SEED>) and never
# trains AR. A compatible completed MDLM is reused; otherwise MDLM is the only
# additional model trained, because paper-style BD3-LM cannot start without it.
#
# PREREQUISITE -- unlike the AR job, BD3-LM does not start from scratch. Its
# config sets require_pretrained, so a TinyStories MDLM checkpoint produced by
# this repository must exist before the BD3 stage. By default the job looks for
#   runs/mdlm_tinystories_seed<SEED>/checkpoints/last.pt
# and verifies dataset, tokenizer, backbone tensor shapes, EMA, and completion.
# If it is absent/incomplete the job trains or resumes it. If it is incompatible,
# it is preserved and a safe architecture-specific MDLM run is used instead.
# Set AUTO_TRAIN_MDLM=0 to restore fail-fast behavior, or point
# BD3_PRETRAIN_CHECKPOINT somewhere else. A resumed BD3 run reads its own
# checkpoint and does not need the MDLM again.
#
# TinyStories is GPT-2 BPE tokenized (50,257 ids; mask/model-only AR BOS 50257;
# tokenizer BOS/EOS <|endoftext|> is 50256). Preparation needs `tiktoken` plus one download of the
# BPE files into TIKTOKEN_CACHE_DIR; training, evaluation, and detokenization
# then read the id -> bytes table out of data/tinystories/meta.pkl and need
# neither.
#
# BD3-LM concatenates [noisy, clean] along the sequence, so with the GPT-2
# vocabulary it needs roughly twice the activation memory of AR/MDLM per
# sequence. configs/tinystories/base.yaml therefore uses a 16x8
# micro-batch/accumulation split, reaching the same global batch of 128 that the
# AR and MDLM runs reach with 32x4 -- the CFM reference's batch.
#
# Validation additionally measures Gen-PPL (GPT-2 Large) and MAUVE on 256
# samples at 64 total BD3 NFEs (1/2/4/8 steps per block for B=4/8/16/32); the
# final evaluation re-measures both with the paper first-hitting sampler.
#
# Tunables (via --export=ALL,KEY=VALUE):
#   SEED                     run seed and name suffix (default 12345)
#   BLOCK_SIZE               4, 8, 16, or 32 (default 16) -> configs/tinystories/bd3lm_b<N>.yaml
#   BD3_PRETRAIN_CHECKPOINT  MDLM checkpoint to initialize from
#   AUTO_TRAIN_MDLM          train/resume MDLM when needed (default 1)
#   MDLM_RUN                 preferred MDLM run name
#   MDLM_COMPAT_RUN          safe fallback name if MDLM_RUN is incompatible
#   MDLM_MAX_STEPS / MDLM_VAL_EVERY / MDLM_BATCH_SIZE / MDLM_GRAD_ACCUM
#                            optional prerequisite-training overrides
#   TINYSTORIES_DATA_DIR     prepared corpus (default data/tinystories)
#   TINYSTORIES_SOURCE_TRAIN / TINYSTORIES_SOURCE_VALID
#                            local TinyStories-*.txt, skipping the ~1.9GB download
#   TINYSTORIES_MAX_TRAIN_BYTES  cap the training text for a smaller run
#   BATCH_SIZE / GRAD_ACCUM  micro-batch and accumulation (keep the product at 128)
#   MAX_STEPS / VAL_EVERY    training length and validation cadence
#   NUM_SAMPLES / EVAL_BATCHES   evaluation sample count and test batches
#   EVAL_CHECKPOINT_NAME     last.pt (default) or best.pt
#   SKIP_COMPLETED=0         re-run even when evaluation.json already exists
#   RUN_TIMESTAMP            timestamp in the default run name (default: job start,
#                            YYYYmmdd_HHMMSS); reuse it to resume the same run
#   RUN_NAME                 override the complete run directory name (also the
#                            safest way to resume a particular timestamped run)

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$PROJECT_DIR/pyproject.toml" || ! -d "$PROJECT_DIR/bcfm_baselines" ]]; then
  if [[ -f "$PROJECT_DIR/BCFM Baselines/pyproject.toml" && -d "$PROJECT_DIR/BCFM Baselines/bcfm_baselines" ]]; then
    PROJECT_DIR="$PROJECT_DIR/BCFM Baselines"
  else
    echo "ERROR: PROJECT_DIR is not the BCFM Baselines repository: $PROJECT_DIR" >&2
    echo "Submit from that directory or export PROJECT_DIR=/absolute/path/to/BCFM Baselines." >&2
    exit 2
  fi
fi
cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Keep the GPT-2 BPE and the Gen-PPL / MAUVE judge weights inside the project so
# compute nodes never need $HOME or the network after the first job.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$TIKTOKEN_CACHE_DIR" "$HF_HOME"

SEED="${SEED:-12345}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TINYSTORIES_DATA_DIR="${TINYSTORIES_DATA_DIR:-data/tinystories}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TIMESTAMP}_bd3lm_b${BLOCK_SIZE}_tinystories_seed${SEED}}"
CONFIG="${CONFIG:-configs/tinystories/bd3lm_b${BLOCK_SIZE}.yaml}"
MDLM_RUN="${MDLM_RUN:-mdlm_tinystories_seed${SEED}}"
MDLM_CONFIG="${MDLM_CONFIG:-configs/tinystories/mdlm.yaml}"
MDLM_COMPAT_RUN="${MDLM_COMPAT_RUN:-mdlm_tinystories_gpt2_h384_l6_seed${SEED}}"
AUTO_TRAIN_MDLM="${AUTO_TRAIN_MDLM:-1}"
if [[ -n "${BD3_PRETRAIN_CHECKPOINT:-}" ]]; then
  BD3_PRETRAIN_WAS_EXPLICIT=1
else
  BD3_PRETRAIN_WAS_EXPLICIT=0
  BD3_PRETRAIN_CHECKPOINT="$PROJECT_DIR/runs/$MDLM_RUN/checkpoints/last.pt"
fi
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/${EVAL_CHECKPOINT_NAME:-last.pt}"
mkdir -p "$RUN_DIR"
# Slurm's #SBATCH output is resolved before RUN_NAME exists. Tee a second,
# self-contained log into the timestamped run for tailing and later provenance.
exec > >(tee -a "$RUN_DIR/job_${SLURM_JOB_ID:-local}.log") 2>&1
echo "Run directory: $RUN_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: no config at $CONFIG (BLOCK_SIZE=$BLOCK_SIZE)." >&2
  echo "Available: $(ls configs/tinystories/bd3lm_b*.yaml | tr '\n' ' ')" >&2
  exit 2
fi
if [[ ! -f "$MDLM_CONFIG" ]]; then
  echo "ERROR: no MDLM prerequisite config at $MDLM_CONFIG." >&2
  exit 2
fi
if [[ "$AUTO_TRAIN_MDLM" != "0" && "$AUTO_TRAIN_MDLM" != "1" ]]; then
  echo "ERROR: AUTO_TRAIN_MDLM must be 0 or 1, got $AUTO_TRAIN_MDLM." >&2
  exit 2
fi
if [[ "$SKIP_COMPLETED" == "1" && -f "$RUN_DIR/evaluation.json" ]]; then
  echo "== $RUN_NAME is already evaluated; nothing to do =="
  exit 0
fi

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
# 1. PRECHECKS
# ===========================================================================
# The BD3-LM mask sampler used to fail only after minutes of training, when a
# block whose sampled rate is epsilon had to contain a masked token. Catch a
# stale checkout here instead, on the GPT-2 vocabulary this job actually trains.
"$PYTHON" - <<'PY'
import torch
from bcfm_baselines.models.bd3lm import BlockDiffusionTextModel

backbone = dict(
    vocab_size=50258, max_sequence_length=256, num_layers=1, hidden_size=32,
    num_attention_heads=4, intermediate_size=64, time_condition_size=16, dropout=0.0,
)
model = BlockDiffusionTextModel(
    backbone=backbone, block_size=16, data_vocab_size=50257, mask_token_id=50257
)
bounds = (float(model.mask_rate_min.item()), float(model.mask_rate_max.item()))
clean = torch.randint(0, 50257, (64, 256))
valid = torch.ones_like(clean, dtype=torch.bool)
torch.manual_seed(0)
for _ in range(20):
    block_t = model._sample_block_times(64, 16, clean.device, bounds)
    token_t = block_t.clamp(min=model.min_time, max=1.0).repeat_interleave(16, dim=1)
    model._sample_mask(clean, valid, token_t, bounds, None)
print("BD3-LM mask sampler check: ok")
PY

# Catch the former delayed-validation crash before training: research sampling
# is first-hitting, but the cheap online curve must explicitly use ancestral
# sampling with a block-size-specific 64-NFE budget.
"$PYTHON" - "$CONFIG" <<'PY'
import sys
from bcfm_baselines.config import load_config
from bcfm_baselines.generative_eval import _generation_config, resolve_config

config = load_config(sys.argv[1])
settings = resolve_config(config)
generation = _generation_config(config["sampling"], settings, batch_size=1, seed=0)
block_size = int(config["model"]["block_size"])
blocks = int(config["sampling"]["length"]) // block_size
if generation.first_hitting:
    raise SystemExit("ERROR: online generative validation must disable first_hitting")
if generation.steps_per_block * blocks != 64:
    raise SystemExit(
        f"ERROR: online BD3 validation budget is {generation.steps_per_block * blocks} NFEs, expected 64"
    )
if not config["sampling"]["first_hitting"]:
    raise SystemExit("ERROR: offline research sampling must enable first_hitting")
print(
    f"BD3-LM validation sampler check: ok "
    f"({generation.steps_per_block} steps/block, 64 NFEs)"
)
PY

# Return 0 for a compatible completed MDLM, 4 for a compatible partial run that
# should be resumed, and 5 for a wrong/corrupt checkpoint. This instantiates the
# actual target BD3 backbone, so the precheck and the trainer cannot disagree.
check_mdlm_checkpoint() {
  local checkpoint_path="$1"
  "$PYTHON" - "$checkpoint_path" "$CONFIG" <<'PY'
import sys
import torch

from bcfm_baselines.config import load_config
from bcfm_baselines.factory import create_model
from bcfm_baselines.train import _load_pretrained_backbone

path, config_path = sys.argv[1:]
target_config = load_config(config_path)
target_model = create_model(target_config)
try:
    metadata = _load_pretrained_backbone(
        target_model,
        path,
        torch.device("cpu"),
        require_ema=True,
        expected_config=target_config,
    )
except Exception as error:
    print(f"MDLM checkpoint is not compatible:\n{error}", file=sys.stderr)
    raise SystemExit(5)

step = metadata["checkpoint_step"]
required = metadata["checkpoint_max_steps"]
if required is None:
    print(f"MDLM checkpoint has no configured max_steps and cannot prove completion: {path}", file=sys.stderr)
    raise SystemExit(5)
if step < required:
    print(f"MDLM checkpoint is compatible but incomplete: step={step}/{required}: {path}")
    raise SystemExit(4)
print(
    f"MDLM checkpoint check: compatible and complete "
    f"(dataset={metadata['source_dataset']}, step={step}/{required}, weights={metadata['weights']})"
)
PY
}

NEEDS_MDLM_PRETRAIN=0
MDLM_TRAIN_RUN_DIR=""

# A resumed BD3 run carries its own weights; only a fresh run needs the MDLM.
if [[ -f "$RUN_DIR/checkpoints/last.pt" ]]; then
  echo "== resuming $RUN_NAME from its own checkpoint; MDLM not needed =="
  unset BD3_PRETRAIN_CHECKPOINT
else
  if [[ -f "$BD3_PRETRAIN_CHECKPOINT" ]]; then
    if check_mdlm_checkpoint "$BD3_PRETRAIN_CHECKPOINT"; then
      MDLM_STATUS=0
    else
      MDLM_STATUS=$?
    fi
  else
    echo "MDLM checkpoint is missing: $BD3_PRETRAIN_CHECKPOINT"
    MDLM_STATUS=3
  fi

  if [[ "$MDLM_STATUS" == "0" ]]; then
    : # Reuse the verified checkpoint.
  elif [[ "$AUTO_TRAIN_MDLM" == "0" ]]; then
    echo "ERROR: no compatible completed TinyStories MDLM is available." >&2
    echo "Set AUTO_TRAIN_MDLM=1 (default) to build it automatically." >&2
    exit 3
  elif [[ "$MDLM_STATUS" == "4" ]]; then
    MDLM_TRAIN_RUN_DIR="$(dirname "$(dirname "$BD3_PRETRAIN_CHECKPOINT")")"
    BD3_PRETRAIN_CHECKPOINT="$MDLM_TRAIN_RUN_DIR/checkpoints/last.pt"
    NEEDS_MDLM_PRETRAIN=1
    echo "== compatible MDLM is partial; it will be resumed in $MDLM_TRAIN_RUN_DIR =="
  else
    # Do not overwrite an explicitly selected or existing incompatible run.
    # Use a stable architecture-specific fallback so an interrupted rebuild can
    # be resumed by submitting this same script again.
    if [[ "$BD3_PRETRAIN_WAS_EXPLICIT" == "1" || -f "$BD3_PRETRAIN_CHECKPOINT" ]]; then
      echo "== preserving incompatible/missing selected checkpoint; using safe MDLM fallback =="
      MDLM_TRAIN_RUN_DIR="$PROJECT_DIR/runs/$MDLM_COMPAT_RUN"
      BD3_PRETRAIN_CHECKPOINT="$MDLM_TRAIN_RUN_DIR/checkpoints/last.pt"
    else
      MDLM_TRAIN_RUN_DIR="$PROJECT_DIR/runs/$MDLM_RUN"
    fi

    if [[ -f "$BD3_PRETRAIN_CHECKPOINT" ]]; then
      if check_mdlm_checkpoint "$BD3_PRETRAIN_CHECKPOINT"; then
        MDLM_STATUS=0
      else
        MDLM_STATUS=$?
      fi
      if [[ "$MDLM_STATUS" == "0" ]]; then
        NEEDS_MDLM_PRETRAIN=0
      elif [[ "$MDLM_STATUS" == "4" ]]; then
        NEEDS_MDLM_PRETRAIN=1
      else
        echo "ERROR: safe MDLM fallback is also incompatible; refusing to overwrite it:" >&2
        echo "  $BD3_PRETRAIN_CHECKPOINT" >&2
        echo "Choose a new MDLM_COMPAT_RUN name and resubmit." >&2
        exit 3
      fi
    else
      NEEDS_MDLM_PRETRAIN=1
    fi
  fi

  if [[ "$NEEDS_MDLM_PRETRAIN" == "1" ]]; then
    echo "== no completed compatible TinyStories MDLM; prerequisite will be trained first =="
    echo "MDLM run directory: $MDLM_TRAIN_RUN_DIR"
  else
    export BD3_PRETRAIN_CHECKPOINT
    echo "== initializing BD3-LM from $BD3_PRETRAIN_CHECKPOINT =="
  fi
fi

# ===========================================================================
# 2. DATA  (re-encodes automatically if the bins predate the GPT-2 tokenizer)
# ===========================================================================
echo "== preparing TinyStories (gpt2) -> $TINYSTORIES_DATA_DIR =="
prep_args=(--output-dir "$TINYSTORIES_DATA_DIR")
[[ -n "${TINYSTORIES_SOURCE_TRAIN:-}" ]] && prep_args+=(--source-train "$TINYSTORIES_SOURCE_TRAIN")
[[ -n "${TINYSTORIES_SOURCE_VALID:-}" ]] && prep_args+=(--source-valid "$TINYSTORIES_SOURCE_VALID")
[[ -n "${TINYSTORIES_MAX_TRAIN_BYTES:-}" ]] && prep_args+=(--max-train-bytes "$TINYSTORIES_MAX_TRAIN_BYTES")
srun --unbuffered "$PYTHON" -m bcfm_baselines.data.prepare_tinystories "${prep_args[@]}"

if [[ "$NEEDS_MDLM_PRETRAIN" == "1" ]]; then
  mkdir -p "$MDLM_TRAIN_RUN_DIR"
  mdlm_overrides=(--set "training.seed=$SEED" --set "dataset.data_dir=$TINYSTORIES_DATA_DIR")
  [[ -n "${MDLM_BATCH_SIZE:-}" ]] && mdlm_overrides+=(--set "training.batch_size=$MDLM_BATCH_SIZE")
  [[ -n "${MDLM_GRAD_ACCUM:-}" ]] && mdlm_overrides+=(--set "training.gradient_accumulation_steps=$MDLM_GRAD_ACCUM")
  [[ -n "${MDLM_MAX_STEPS:-}" ]] && mdlm_overrides+=(--set "training.max_steps=$MDLM_MAX_STEPS")
  [[ -n "${MDLM_VAL_EVERY:-}" ]] && mdlm_overrides+=(--set "training.validation_every=$MDLM_VAL_EVERY")
  echo "== training/resuming TinyStories MDLM prerequisite ($MDLM_CONFIG) =="
  env -u BD3_PRETRAIN_CHECKPOINT srun --unbuffered "$PYTHON" -m bcfm_baselines.train \
    --config "$MDLM_CONFIG" --run-dir "$MDLM_TRAIN_RUN_DIR" --resume auto \
    "${mdlm_overrides[@]}"
  if ! check_mdlm_checkpoint "$BD3_PRETRAIN_CHECKPOINT"; then
    echo "ERROR: MDLM training returned without a compatible completed checkpoint." >&2
    exit 3
  fi
  export BD3_PRETRAIN_CHECKPOINT
  echo "== MDLM prerequisite complete; initializing BD3-LM from $BD3_PRETRAIN_CHECKPOINT =="
fi

# ===========================================================================
# 3. TRAIN + EVALUATE  (BD3-LM only)
# ===========================================================================
mkdir -p "$RUN_DIR"

overrides=()
[[ -n "${BATCH_SIZE:-}" ]] && overrides+=(--set "training.batch_size=$BATCH_SIZE")
[[ -n "${GRAD_ACCUM:-}" ]] && overrides+=(--set "training.gradient_accumulation_steps=$GRAD_ACCUM")
[[ -n "${MAX_STEPS:-}" ]] && overrides+=(--set "training.max_steps=$MAX_STEPS")
[[ -n "${VAL_EVERY:-}" ]] && overrides+=(--set "training.validation_every=$VAL_EVERY")
overrides+=(--set "training.seed=$SEED")
overrides+=(--set "dataset.data_dir=$TINYSTORIES_DATA_DIR")

echo "== training $RUN_NAME ($CONFIG, block_size=$BLOCK_SIZE) =="
srun --unbuffered "$PYTHON" -m bcfm_baselines.train \
  --config "$CONFIG" --run-dir "$RUN_DIR" --resume auto \
  ${overrides[@]+"${overrides[@]}"}

echo "== evaluating $RUN_NAME =="
srun --unbuffered "$PYTHON" -m bcfm_baselines.evaluate \
  --checkpoint "$CHECKPOINT" --output "$RUN_DIR/evaluation.json" \
  --eval-batch-size "$EVAL_BATCH_SIZE" --eval-batches "$EVAL_BATCHES" \
  --num-samples "$NUM_SAMPLES"

echo "== done: $RUN_DIR/evaluation.json =="
