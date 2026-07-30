#!/usr/bin/env bash
# Evaluate colleague AR / MDLM / BD3-LM Text8 checkpoints with GPT-J gen-PPL
# (same harness as M1/M2/M3) and rebuild aggregate plots.
#
# Usage:
#   bash scripts/run_baselines_nfe.sh
#   SEEDS="0 1 2" N_SAMPLES=512 EVAL_GPU=1 bash scripts/run_baselines_nfe.sh
#   ONLY_AGGREGATE=1 bash scripts/run_baselines_nfe.sh
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
echo "BASELINES_CODE=$BASELINES_CODE"

# Checkpoint root — see docs/baselines_eval_guide.md
BASE_RUNS="${BASE_RUNS:-$PWD/checkpoints/baselines}"
AR_CKPT="${AR_CKPT:-$BASE_RUNS/ar_text8_seed12345/checkpoints/last.pt}"
MDLM_CKPT="${MDLM_CKPT:-$BASE_RUNS/mdlm_text8_seed12345/checkpoints/last.pt}"
BD3_CKPT="${BD3_CKPT:-$BASE_RUNS/bd3lm_b16_text8_seed12345/checkpoints/last.pt}"

SEEDS="${SEEDS:-0 1 2}"
N_SAMPLES="${N_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LENGTH="${LENGTH:-256}"
JUDGE_MODEL="${JUDGE_MODEL:-EleutherAI/gpt-j-6b}"
JUDGE_BS="${JUDGE_BS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ONLY_AGGREGATE="${ONLY_AGGREGATE:-0}"
RESULT_TAG="${RESULT_TAG:-base}"

# Sweeps (aligned with colleague docs / our M3 B=16 sweep)
MDLM_STEPS="${MDLM_STEPS:-16 32 64 128}"
BD3_STEPS="${BD3_STEPS:-1 2 4 8 16}"
RUN_AR="${RUN_AR:-1}"
RUN_MDLM="${RUN_MDLM:-1}"
RUN_BD3="${RUN_BD3:-1}"
# Official BD3 first-hitting is slow (needs steps_per_block >= B); default off for NFE sweep.
BD3_FIRST_HITTING="${BD3_FIRST_HITTING:-0}"

run_one() {
  local exp="$1"; shift
  local out="results/${exp}/metrics.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "skip $exp (exists)"
    return 0
  fi
  echo "================================================================"
  echo ">>> $exp"
  echo "    $*"
  echo "================================================================"
  python -m eval.run_baseline_eval --exp-name "$exp" "$@"
}

mkdir -p results

if [[ "$ONLY_AGGREGATE" != "1" ]]; then
  for seed in $SEEDS; do
    if [[ "$RUN_AR" == "1" ]]; then
      run_one "${RESULT_TAG}_ar_L${LENGTH}_seed${seed}" \
        --checkpoint "$AR_CKPT" --seed "$seed" \
        --n-samples "$N_SAMPLES" --batch-size "$BATCH_SIZE" --length "$LENGTH" \
        --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
    fi

    if [[ "$RUN_MDLM" == "1" ]]; then
      for s in $MDLM_STEPS; do
        run_one "${RESULT_TAG}_mdlm_s${s}_L${LENGTH}_seed${seed}" \
          --checkpoint "$MDLM_CKPT" --seed "$seed" --num-steps "$s" \
          --n-samples "$N_SAMPLES" --batch-size "$BATCH_SIZE" --length "$LENGTH" \
          --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
      done
    fi

    if [[ "$RUN_BD3" == "1" ]]; then
      fh_args=(--no-first-hitting)
      fh_tag="nofh"
      if [[ "$BD3_FIRST_HITTING" == "1" ]]; then
        fh_args=(--first-hitting)
        fh_tag="fh"
      fi
      for s in $BD3_STEPS; do
        run_one "${RESULT_TAG}_bd3_B16_s${s}_${fh_tag}_L${LENGTH}_seed${seed}" \
          --checkpoint "$BD3_CKPT" --seed "$seed" --steps-per-block "$s" \
          "${fh_args[@]}" \
          --n-samples "$N_SAMPLES" --batch-size "$BATCH_SIZE" --length "$LENGTH" \
          --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
      done
    fi
  done
fi

echo "=== aggregate + figures ==="
python -m eval.aggregate
echo "Done. CSV: results/summary.csv  Figs: results/figures/"
