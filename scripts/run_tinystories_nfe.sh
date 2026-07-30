#!/usr/bin/env bash
# TinyStories: M1/M2 (NEW.zip semicat) + AR/MDLM (colleague baselines), GPT-J gen-PPL.
#
# Usage:
#   bash scripts/run_tinystories_nfe.sh
#   EVAL_GPU=1 SEEDS="0 1 2" bash scripts/run_tinystories_nfe.sh
#   ONLY_AGGREGATE=1 bash scripts/run_tinystories_nfe.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$PWD/data}"
export BASELINES_CODE="${BASELINES_CODE:-$PWD/baselines}"
export TEXT8_DATA_DIR="${TEXT8_DATA_DIR:-$PWD/data/tinystories}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}"
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "BASELINES_CODE=$BASELINES_CODE"

# Checkpoint root — see docs/baselines_eval_guide.md
BASE_RUNS="${BASE_RUNS:-$PWD/checkpoints/baselines}"
M1_CKPT="${M1_CKPT:-$BASE_RUNS/tinystories_a100/checkpoints/best.ckpt}"
AR_CKPT="${AR_CKPT:-$BASE_RUNS/ar_tinystories_seed12345/checkpoints/last.pt}"
MDLM_CKPT="${MDLM_CKPT:-$BASE_RUNS/mdlm_tinystories_seed12345/checkpoints/last.pt}"
TS_DATA="${TS_DATA:-$PWD/data/tinystories}"

SEEDS="${SEEDS:-0 1 2}"
N_SAMPLES="${N_SAMPLES:-512}"
# GPT-2 vocab CFM is heavier; keep micro-batch modest
M1_BATCH="${M1_BATCH:-32}"
BASE_BATCH="${BASE_BATCH:-128}"
LENGTH="${LENGTH:-256}"
JUDGE_MODEL="${JUDGE_MODEL:-EleutherAI/gpt-j-6b}"
JUDGE_BS="${JUDGE_BS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ONLY_AGGREGATE="${ONLY_AGGREGATE:-0}"
RESULT_TAG="${RESULT_TAG:-ts}"

M1_STEPS="${M1_STEPS:-1 2 4 8 16}"
M2_B="${M2_B:-16}"
M2_STEPS="${M2_STEPS:-1 2 4 8 16}"
MDLM_STEPS="${MDLM_STEPS:-1 2 4 8 16}"
RUN_M1="${RUN_M1:-1}"
RUN_M2="${RUN_M2:-1}"
RUN_AR="${RUN_AR:-1}"
RUN_MDLM="${RUN_MDLM:-1}"

if [[ "${RUN_AR}" == "1" || "${RUN_MDLM}" == "1" ]]; then
  if [[ ! -f "$TS_DATA/meta.pkl" ]]; then
    echo "Preparing TinyStories byte corpus at $TS_DATA ..."
    PYTHONPATH="$BASELINES_CODE" python -m bcfm_baselines.data.prepare_tinystories \
      --output-dir "$TS_DATA"
  fi
fi

run_cfm() {
  local exp="$1"; shift
  local out="results/${exp}/metrics.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "skip $exp (exists)"
    return 0
  fi
  echo "================================================================"
  echo ">>> $exp"
  echo "================================================================"
  python -m eval.run_tinystories_cfm_eval --exp-name "$exp" "$@"
}

run_base() {
  local exp="$1"; shift
  local out="results/${exp}/metrics.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "skip $exp (exists)"
    return 0
  fi
  echo "================================================================"
  echo ">>> $exp"
  echo "================================================================"
  python -m eval.run_baseline_eval --exp-name "$exp" \
    --text8-data-dir "$TS_DATA" "$@"
}

mkdir -p results

if [[ "$ONLY_AGGREGATE" != "1" ]]; then
  for seed in $SEEDS; do
    if [[ "$RUN_M1" == "1" ]]; then
      for s in $M1_STEPS; do
        run_cfm "${RESULT_TAG}_m1_full_L${LENGTH}_s${s}_seed${seed}" \
          --checkpoint "$M1_CKPT" --sampler full_cfm --steps "$s" \
          --model-id M1-TS --seed "$seed" \
          --n-samples "$N_SAMPLES" --batch-size "$M1_BATCH" --length "$LENGTH" \
          --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
      done
    fi

    if [[ "$RUN_M2" == "1" ]]; then
      for B in $M2_B; do
        for s in $M2_STEPS; do
          run_cfm "${RESULT_TAG}_m2_kv_B${B}_s${s}_L${LENGTH}_seed${seed}" \
            --checkpoint "$M1_CKPT" --sampler bcfm_infer \
            --block-size "$B" --steps-per-block "$s" \
            --model-id M2-TS --seed "$seed" \
            --n-samples "$N_SAMPLES" --batch-size "$M1_BATCH" --length "$LENGTH" \
            --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
        done
      done
    fi

    if [[ "$RUN_AR" == "1" ]]; then
      run_base "${RESULT_TAG}_ar_L${LENGTH}_seed${seed}" \
        --checkpoint "$AR_CKPT" --seed "$seed" \
        --n-samples "$N_SAMPLES" --batch-size "$BASE_BATCH" --length "$LENGTH" \
        --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
      # rename model tag in metrics to AR-TS for plot separation from Text8 AR
      python - <<PY
import json
from pathlib import Path
p = Path("results/${RESULT_TAG}_ar_L${LENGTH}_seed${seed}/metrics.json")
m = json.loads(p.read_text()); m["model"] = "AR-TS"; m["dataset"] = "tinystories"
p.write_text(json.dumps(m, indent=2))
PY
    fi

    if [[ "$RUN_MDLM" == "1" ]]; then
      for s in $MDLM_STEPS; do
        run_base "${RESULT_TAG}_mdlm_s${s}_L${LENGTH}_seed${seed}" \
          --checkpoint "$MDLM_CKPT" --seed "$seed" --num-steps "$s" \
          --n-samples "$N_SAMPLES" --batch-size "$BASE_BATCH" --length "$LENGTH" \
          --judge --judge-model "$JUDGE_MODEL" --judge-batch-size "$JUDGE_BS"
        python - <<PY
import json
from pathlib import Path
p = Path("results/${RESULT_TAG}_mdlm_s${s}_L${LENGTH}_seed${seed}/metrics.json")
m = json.loads(p.read_text()); m["model"] = "MDLM-TS"; m["dataset"] = "tinystories"
p.write_text(json.dumps(m, indent=2))
PY
      done
    fi
  done
fi

python -m eval.aggregate
echo "Done. CSV: results/summary.csv  Figs: results/figures/"
