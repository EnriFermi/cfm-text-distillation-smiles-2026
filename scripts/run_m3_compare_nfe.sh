#!/usr/bin/env bash
# Evaluate M3 (block-trained BlockSemicatModule + duo.DIT) and compare to
# existing M1/M2 curves in results/ (from the baseline CFM ckpt).
#
# This ckpt is NOT BlockDIT — sample with sampler=bcfm_infer (clean KV cache).
# Trained block_size=16; sweep steps_per_block.
#
# Usage:
#   bash scripts/run_m3_compare_nfe.sh
#   M3_STEPS="1 2 4 8" SEEDS="0 1 2" bash scripts/run_m3_compare_nfe.sh
#   ONLY_AGGREGATE=1 bash scripts/run_m3_compare_nfe.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$PWD/data}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-2}"
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

M3_CKPT="${M3_CKPT:-/home/zolotovskijal/checkpoints/step_0090000.ckpt}"
SEEDS="${SEEDS:-0 1 2}"
N_SAMPLES="${N_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LENGTH="${LENGTH:-256}"
REPORT_B="${REPORT_B:-16}"
M3_B="${M3_B:-16}"                      # must match training block_size
M3_STEPS="${M3_STEPS:-1 2 4 8 16}"
JUDGE="${JUDGE:-true}"
JUDGE_MODEL="${JUDGE_MODEL:-EleutherAI/gpt-j-6b}"
JUDGE_BS="${JUDGE_BS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ONLY_AGGREGATE="${ONLY_AGGREGATE:-0}"
RESULT_TAG="${RESULT_TAG:-m3cmp}"

if [[ ! -f "$M3_CKPT" ]]; then
  echo "ERROR: M3 checkpoint not found: $M3_CKPT" >&2
  exit 1
fi

COMMON=(
  ckpt_path="$M3_CKPT"
  model=block_duo_text8
  "model.block_size=$M3_B"
  data.k="$LENGTH"
  model.net.length="$LENGTH"
  judge="$JUDGE"
  detok=true
  judge_model="$JUDGE_MODEL"
  judge_batch_size="$JUDGE_BS"
  n_samples="$N_SAMPLES"
  batch_size="$BATCH_SIZE"
  report_block_size="$REPORT_B"
)

run_one() {
  local exp="$1" seed="$2"; shift 2
  local out="results/${exp}/metrics.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "skip $exp (exists)"
    return 0
  fi
  echo "================================================================"
  echo ">>> $exp"
  echo "    $*"
  echo "================================================================"
  python -m eval.run_eval "$@" "${COMMON[@]}" seed="$seed" exp_name="$exp"
}

mkdir -p results

if [[ "$ONLY_AGGREGATE" != "1" ]]; then
  echo "=== M3 block-trained (B=$M3_B × steps/block) ==="
  for seed in $SEEDS; do
    for s in $M3_STEPS; do
      run_one "${RESULT_TAG}_m3_kv_B${M3_B}_s${s}_L${LENGTH}_seed${seed}" "$seed" \
        sampler=bcfm_infer \
        "sampler.block_size=$M3_B" "sampler.steps_per_block=$s" \
        model_id=M3
    done
  done
fi

echo "=== aggregate + figures (includes existing M1/M2 if present) ==="
python -m eval.aggregate

echo
echo "Done."
echo "  M3 ckpt: $M3_CKPT"
echo "  CSV:     results/summary.csv"
echo "  Figs:    results/figures/"
