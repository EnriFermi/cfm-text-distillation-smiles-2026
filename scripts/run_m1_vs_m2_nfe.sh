#!/usr/bin/env bash
# M1 (full CFM) vs M2 (block + clean-prefix KV cache): gen-PPL / entropy vs
# all network forwards. M2 NFE includes both denoising and useful cache builds.
#
# Usage:
#   bash scripts/run_m1_vs_m2_nfe.sh
#   CKPT=/path/to.ckpt SEEDS="0 1 2" bash scripts/run_m1_vs_m2_nfe.sh
#   SKIP_EXISTING=0 bash scripts/run_m1_vs_m2_nfe.sh   # re-run everything
#   JUDGE=false bash scripts/run_m1_vs_m2_nfe.sh        # timing/entropy only
#   ONLY_AGGREGATE=1 bash scripts/run_m1_vs_m2_nfe.sh   # just rebuild plots
#
# Writes results/<exp>/metrics.json, then:
#   results/summary.csv
#   results/figures/nfe_total_vs_genppl.png
#   results/figures/nfe_total_vs_nll.png
#   results/figures/sequence_latency.png
#   results/figures/tokens_per_sec.png
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$PWD/data}"

# GPT-J-6B fp16 needs ~12GB; pick a free GPU unless the user already set one.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-2}"
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CKPT="${CKPT:-/home/zolotovskijal/checkpoints/last.ckpt}"
SEEDS="${SEEDS:-0 1 2}"
N_SAMPLES="${N_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LENGTH="${LENGTH:-256}"                 # total tokens (= data.k); must match ckpt RoPE train length for fair PPL
REPORT_B="${REPORT_B:-16}"              # entropy bucket size (shared across M1/M2 curves)
# Defaults match /home/zolotovskijal/checkpoints/last.ckpt (naive + lag).
# Override if your ckpt differs (e.g. EMBED_TYPE=rms SD_TYPE=ecld).
EMBED_TYPE="${EMBED_TYPE:-naive}"
SD_TYPE="${SD_TYPE:-lag}"
SD_PROP="${SD_PROP:-0.25}"
JUDGE="${JUDGE:-true}"                  # false => skip gen_ppl (faster smoke)
JUDGE_MODEL="${JUDGE_MODEL:-EleutherAI/gpt-j-6b}"  # or gpt2-large
JUDGE_BS="${JUDGE_BS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ONLY_AGGREGATE="${ONLY_AGGREGATE:-0}"
RUN_GOLD="${RUN_GOLD:-1}"
RESULT_TAG="${RESULT_TAG:-nfeall_v2}"

# Sweeps (space-separated)
M1_STEPS="${M1_STEPS:-1 2 4 8 16}"
M2_B="${M2_B:-16 32}"
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
  if [[ "$RUN_GOLD" == "1" ]]; then
    run_one "${RESULT_TAG}_gold_L${LENGTH}_seed0" 0 \
      sampler=gold model_id=gold
  fi

  echo "=== M1 full-sequence CFM (steps sweep) ==="
  for seed in $SEEDS; do
    for s in $M1_STEPS; do
      run_one "${RESULT_TAG}_m1_full_L${LENGTH}_s${s}_seed${seed}" "$seed" \
        sampler=full_cfm "sampler.steps=$s" model_id=M1
    done
  done

  echo "=== M2 block + clean KV-cache (B × steps/block) ==="
  for seed in $SEEDS; do
    for B in $M2_B; do
      for s in $M2_STEPS; do
        run_one "${RESULT_TAG}_m2_kv_B${B}_s${s}_L${LENGTH}_seed${seed}" "$seed" \
          sampler=bcfm_infer \
          "sampler.block_size=$B" "sampler.steps_per_block=$s" \
          model_id=M2
      done
    done
  done
fi

echo "=== aggregate + figures ==="
python -m eval.aggregate

echo
echo "Done."
echo "  CSV:    results/summary.csv"
echo "  Figs:   results/figures/nfe_total_vs_genppl.png"
echo "          results/figures/nfe_total_vs_nll.png"
echo "          results/figures/sequence_latency.png"
echo "          results/figures/tokens_per_sec.png"
echo "  Tip:    ONLY_AGGREGATE=1 bash scripts/run_m1_vs_m2_nfe.sh   # replot only"
