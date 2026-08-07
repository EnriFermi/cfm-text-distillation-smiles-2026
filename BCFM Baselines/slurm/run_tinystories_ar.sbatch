#!/bin/bash
#SBATCH --job-name=bcfm-ts-ar
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# Cluster-specific: uncomment / adjust for your site, or export SBATCH_PARTITION / SBATCH_ACCOUNT.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --requeue

# Single-run job: the AR baseline on TinyStories, nothing else.
#
# Unlike run_all.sbatch / run_remaining.sbatch this job touches exactly one
# timestamped run directory
# (runs/<YYYYmmdd_HHMMSS>_ar_tinystories_seed<SEED>) and never trains MDLM or
# BD3-LM, so it can be submitted independently of the multi-model jobs.
#
# TinyStories is GPT-2 BPE tokenized (50,257 ids; mask/model-only AR BOS 50257;
# tokenizer BOS/EOS <|endoftext|> is 50256). Preparation needs `tiktoken` plus one download of the
# BPE files into TIKTOKEN_CACHE_DIR; training, evaluation, and detokenization
# then read the id -> bytes table out of data/tinystories/meta.pkl and need
# neither. If data/tinystories was built by the older byte-level pipeline, this
# job re-encodes it in place -- prepare_tinystories detects the tokenizer written
# into dataset_metadata.json.
#
# The GPT-2 vocabulary makes the logits (B x 256 x 50258) the largest activation
# in the step (~0.12GB per sequence in bf16 at the 384-wide backbone), so
# this A100 job defaults to a 128x1 micro-batch/accumulation split for a global
# batch of 128 -- the CFM reference's effective batch and the same 3.28e9-token
# budget over 100k steps, without four small forward/backward launches per
# optimizer update. Override BATCH_SIZE/GRAD_ACCUM if the target GPU is smaller.
#
# Every 5K optimizer steps validation additionally measures Gen-PPL (GPT-2
# Large) and MAUVE on 256 samples; the final evaluation re-measures both on 512.
#
# Tunables (via --export=ALL,KEY=VALUE):
#   SEED                     run seed and name suffix (default 12345)
#   TINYSTORIES_DATA_DIR     prepared corpus (default data/tinystories)
#   TINYSTORIES_SOURCE_TRAIN / TINYSTORIES_SOURCE_VALID
#                            local TinyStories-*.txt, skipping the ~1.9GB download
#   TINYSTORIES_MAX_TRAIN_BYTES  cap the training text for a smaller run
#   BATCH_SIZE / GRAD_ACCUM  micro-batch and accumulation (defaults 128 / 1;
#                            keep the product at 128)
#   MAX_STEPS / VAL_EVERY    training length and validation cadence
#   NUM_SAMPLES / EVAL_BATCHES   evaluation sample count and test batches
#   EVAL_CHECKPOINT_NAME     last.pt (default) or best.pt
#   SKIP_COMPLETED=0         re-run even when evaluation.json already exists
#   RUN_TIMESTAMP            timestamp in the default run name (job start,
#                            YYYYmmdd_HHMMSS)
#   RUN_NAME                 override the complete run name; reuse to resume

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
# Keep the GPT-2 BPE and judge weights inside the project so compute nodes can
# reuse them across validation calls and jobs.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT_DIR/.cache/tiktoken}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
mkdir -p "$TIKTOKEN_CACHE_DIR" "$HF_HOME"

SEED="${SEED:-12345}"
TINYSTORIES_DATA_DIR="${TINYSTORIES_DATA_DIR:-data/tinystories}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TIMESTAMP}_ar_tinystories_seed${SEED}}"
CONFIG="${CONFIG:-configs/tinystories/ar.yaml}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/${EVAL_CHECKPOINT_NAME:-last.pt}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/job_${SLURM_JOB_ID:-local}.log") 2>&1
echo "Run directory: $RUN_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: no AR config at $CONFIG." >&2
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

"$PYTHON" - "$CONFIG" "$BATCH_SIZE" "$GRAD_ACCUM" <<'PY'
import sys

from bcfm_baselines.config import load_config
from bcfm_baselines.generative_eval import require_metric_dependencies, resolve_config

config = load_config(sys.argv[1])
if config["model"]["type"] != "ar":
    raise SystemExit(f"ERROR: AR sbatch requires model.type=ar, got {config['model']['type']!r}")
require_metric_dependencies(resolve_config(config))

# Lock the capacity/training budget to semicat's TinyStories CFM config. AR has
# one necessary extra model-only BOS row, hence backbone vocab 50,258 while the
# shared data/output vocabulary remains GPT-2's 50,257 tokens.
expected = {
    "dataset.name": "tinystories",
    "dataset.tokenizer": "gpt2",
    "dataset.vocabulary_size": 50257,
    "dataset.sequence_length": 256,
    "model.data_vocab_size": 50257,
    "model.bos_token_id": 50257,
    "model.backbone.vocab_size": 50258,
    "model.backbone.max_sequence_length": 256,
    "model.backbone.num_layers": 6,
    "model.backbone.hidden_size": 384,
    "model.backbone.num_attention_heads": 6,
    "model.backbone.intermediate_size": 1536,
    "model.backbone.time_condition_size": 128,
    "model.backbone.dropout": 0.1,
    "model.backbone.rotary_base": 10000.0,
    "training.max_steps": 100000,
    "training.learning_rate": 1e-4,
    "training.weight_decay": 0.1,
    "training.betas": [0.9, 0.999],
    "training.epsilon": 1e-8,
    "training.warmup_steps": 500,
    "training.warmup_start_factor": 1e-3,
    "training.lr_decay": "constant",
    "training.gradient_clip_norm": 1.0,
    "training.precision": "bf16",
}

def nested(path):
    value = config
    for part in path.split("."):
        value = value[part]
    return value

mismatches = [
    f"{path}: got {nested(path)!r}, expected {value!r}"
    for path, value in expected.items()
    if nested(path) != value
]
if mismatches:
    raise SystemExit("ERROR: AR/CFM parity check failed:\n  " + "\n  ".join(mismatches))
global_batch = int(config["training"]["batch_size"]) * int(
    config["training"]["gradient_accumulation_steps"]
)
requested_global_batch = int(sys.argv[2]) * int(sys.argv[3])
if requested_global_batch != global_batch:
    raise SystemExit(
        f"ERROR: BATCH_SIZE*GRAD_ACCUM={requested_global_batch}, expected {global_batch} "
        "for a CFM-matched effective batch"
    )
print(
    f"AR precheck: ok (validation_every={config['training']['validation_every']}, "
    f"micro_batch={sys.argv[2]}, grad_accum={sys.argv[3]}, global_batch={global_batch})"
)
PY

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
# 2. TRAIN + EVALUATE  (AR only)
# ===========================================================================
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
