#!/usr/bin/env bash
# End-to-end latency benchmark for one generated sequence (batch_size=1).
# Updates existing metrics in place; it never regenerates samples or judge-PPL.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$PWD/data}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-2}"
fi

CKPT="${CKPT:-/home/zolotovskijal/checkpoints/last.ckpt}"
RESULT_TAG="${RESULT_TAG:-nfeall_v2}"
SEED="${SEED:-0}"                 # Sampling seed does not affect the fixed-shape latency path.
LENGTH="${LENGTH:-256}"
EMBED_TYPE="${EMBED_TYPE:-naive}"
SD_TYPE="${SD_TYPE:-lag}"
SD_PROP="${SD_PROP:-0.25}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
REPEATS="${REPEATS:-30}"
M1_STEPS="${M1_STEPS:-1 2 4 8 16}"
M2_B="${M2_B:-16 32 64 128}"
M2_STEPS="${M2_STEPS:-1 2 4 8}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

COMMON=(
  ckpt_path="$CKPT"
  data.k="$LENGTH"
  model.net.length="$LENGTH"
  model.net.embed_type="$EMBED_TYPE"
  +model.sd_type="$SD_TYPE"
  +model.sd_prop="$SD_PROP"
  latency_only=true
  latency_warmup_runs="$WARMUP_RUNS"
  latency_repeats="$REPEATS"
  judge=false
  detok=false
  n_samples=1
  batch_size=1
  seed="$SEED"
)

measure_one() {
  local exp="$1"; shift
  if [[ ! -f "results/${exp}/metrics.json" ]]; then
    echo "ERROR: missing results/${exp}/metrics.json" >&2
    exit 1
  fi
  echo ">>> latency: $exp"
  python -m eval.run_eval "$@" "${COMMON[@]}" exp_name="$exp"
}

for s in $M1_STEPS; do
  measure_one "${RESULT_TAG}_m1_full_L${LENGTH}_s${s}_seed${SEED}" \
    sampler=full_cfm "sampler.steps=$s" model_id=M1
done

for B in $M2_B; do
  for s in $M2_STEPS; do
    measure_one "${RESULT_TAG}_m2_kv_B${B}_s${s}_L${LENGTH}_seed${SEED}" \
      sampler=bcfm_infer "sampler.block_size=$B" "sampler.steps_per_block=$s" model_id=M2
  done
done

python -m eval.aggregate
