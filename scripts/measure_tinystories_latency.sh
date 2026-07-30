#!/usr/bin/env bash
# Batch=1 L=256 latency for TinyStories M1/M2/AR/MDLM (updates metrics in place).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export BASELINES_CODE="${BASELINES_CODE:-$PWD/baselines}"
export TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-$PWD/data/tinystories}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}"
fi

# Checkpoint root — see docs/baselines_eval_guide.md
BASE_RUNS="${BASE_RUNS:-$PWD/checkpoints/baselines}"
M1_CKPT="${M1_CKPT:-$BASE_RUNS/tinystories_a100/checkpoints/best.ckpt}"
AR_CKPT="${AR_CKPT:-$BASE_RUNS/ar_tinystories_seed12345/checkpoints/last.pt}"
MDLM_CKPT="${MDLM_CKPT:-$BASE_RUNS/mdlm_tinystories_seed12345/checkpoints/last.pt}"
TS_DATA="${TS_DATA:-$PWD/data/tinystories}"
SEED="${SEED:-0}"
LENGTH="${LENGTH:-256}"
RESULT_TAG="${RESULT_TAG:-ts}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
REPEATS="${REPEATS:-30}"
M1_STEPS="${M1_STEPS:-1 2 4 8 16}"
M2_B="${M2_B:-16}"
M2_STEPS="${M2_STEPS:-1 2 4 8 16}"
MDLM_STEPS="${MDLM_STEPS:-1 2 4 8 16}"

measure_cfm() {
  local exp="$1"; shift
  [[ -f "results/${exp}/metrics.json" ]] || { echo "missing $exp"; exit 1; }
  echo ">>> latency $exp"
  python -m eval.run_tinystories_cfm_eval --exp-name "$exp" --latency-only \
    --latency-warmup-runs "$WARMUP_RUNS" --latency-repeats "$REPEATS" \
    --checkpoint "$M1_CKPT" --length "$LENGTH" --seed "$SEED" "$@"
}

measure_base() {
  local exp="$1"; shift
  [[ -f "results/${exp}/metrics.json" ]] || { echo "missing $exp"; exit 1; }
  echo ">>> latency $exp"
  python -m eval.run_baseline_eval --exp-name "$exp" --latency-only \
    --latency-warmup-runs "$WARMUP_RUNS" --latency-repeats "$REPEATS" \
    --text8-data-dir "$TS_DATA" --length "$LENGTH" --seed "$SEED" "$@"
}

for s in $M1_STEPS; do
  measure_cfm "${RESULT_TAG}_m1_full_L${LENGTH}_s${s}_seed${SEED}" \
    --sampler full_cfm --steps "$s"
done
for B in $M2_B; do
  for s in $M2_STEPS; do
    measure_cfm "${RESULT_TAG}_m2_kv_B${B}_s${s}_L${LENGTH}_seed${SEED}" \
      --sampler bcfm_infer --block-size "$B" --steps-per-block "$s"
  done
done
measure_base "${RESULT_TAG}_ar_L${LENGTH}_seed${SEED}" --checkpoint "$AR_CKPT"
for s in $MDLM_STEPS; do
  measure_base "${RESULT_TAG}_mdlm_s${s}_L${LENGTH}_seed${SEED}" \
    --checkpoint "$MDLM_CKPT" --num-steps "$s"
done

python -m eval.aggregate
echo "Done TinyStories latency."
