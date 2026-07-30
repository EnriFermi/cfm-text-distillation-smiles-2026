#!/usr/bin/env bash
# End-to-end latency for one L=256 sequence (batch_size=1) on colleague baselines.
# Same protocol as scripts/measure_sequence_latency.sh (warmup + synced repeats).
# Updates existing results/base_*/metrics.json in place; does not re-judge.
#
# Usage:
#   bash scripts/measure_baselines_latency.sh
#   EVAL_GPU=1 REPEATS=30 bash scripts/measure_baselines_latency.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$PWD/data}"
export BASELINES_CODE="${BASELINES_CODE:-$PWD/baselines}"
export TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-$PWD/data/text8}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}"
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Checkpoint root — see docs/baselines_eval_guide.md
BASE_RUNS="${BASE_RUNS:-$PWD/checkpoints/baselines}"
AR_CKPT="${AR_CKPT:-$BASE_RUNS/ar_text8_seed12345/checkpoints/last.pt}"
MDLM_CKPT="${MDLM_CKPT:-$BASE_RUNS/mdlm_text8_seed12345/checkpoints/last.pt}"
BD3_CKPT="${BD3_CKPT:-$BASE_RUNS/bd3lm_b16_text8_seed12345/checkpoints/last.pt}"

SEED="${SEED:-0}"
LENGTH="${LENGTH:-256}"
RESULT_TAG="${RESULT_TAG:-base}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
REPEATS="${REPEATS:-30}"
MDLM_STEPS="${MDLM_STEPS:-16 32 64 128}"
BD3_STEPS="${BD3_STEPS:-1 2 4 8 16}"
RUN_AR="${RUN_AR:-1}"
RUN_MDLM="${RUN_MDLM:-1}"
RUN_BD3="${RUN_BD3:-1}"

measure_one() {
  local exp="$1"; shift
  local out="results/${exp}/metrics.json"
  if [[ ! -f "$out" ]]; then
    echo "ERROR: missing $out" >&2
    exit 1
  fi
  echo ">>> latency: $exp"
  python -m eval.run_baseline_eval \
    --exp-name "$exp" \
    --length "$LENGTH" \
    --seed "$SEED" \
    --latency-only \
    --latency-warmup-runs "$WARMUP_RUNS" \
    --latency-repeats "$REPEATS" \
    "$@"
}

if [[ "$RUN_AR" == "1" ]]; then
  measure_one "${RESULT_TAG}_ar_L${LENGTH}_seed${SEED}" --checkpoint "$AR_CKPT"
fi

if [[ "$RUN_MDLM" == "1" ]]; then
  for s in $MDLM_STEPS; do
    measure_one "${RESULT_TAG}_mdlm_s${s}_L${LENGTH}_seed${SEED}" \
      --checkpoint "$MDLM_CKPT" --num-steps "$s"
  done
fi

if [[ "$RUN_BD3" == "1" ]]; then
  for s in $BD3_STEPS; do
    measure_one "${RESULT_TAG}_bd3_B16_s${s}_nofh_L${LENGTH}_seed${SEED}" \
      --checkpoint "$BD3_CKPT" --steps-per-block "$s" --no-first-hitting
  done
fi

python -m eval.aggregate
echo "Done. Updated sequence_latency.png"
