#!/usr/bin/env bash
# Fixed-hyperparameter M1 vs M2 at L=256 (training length): gold + one M1 + one M2 point.
#
# Key CLI knobs (all passed to `python -m eval.run_eval`):
#   n_samples=256     how many sequences to generate and score
#   batch_size=128    GPU batch for sampling (not the eval sample count)
#   data.k=256        sequence length L (default text8; must match model.net.length)
#   seed=12345        RNG for reproducible samples
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PROJECT_ROOT="$PWD"

CKPT=/root/checkpoints/last.ckpt
B=16
STEPS=4

COMMON=(
  ckpt_path="$CKPT"
  model.net.embed_type=naive
  +model.sd_type=lag
  +model.sd_prop=0.25
  judge_model=EleutherAI/gpt-j-6b
  judge_batch_size=4
  n_samples=256
  batch_size=128
  seed=12345
  report_block_size="$B"
)

run_one() {
  echo ">>> $*"
  python -m eval.run_eval "$@" "${COMMON[@]}"
}

rm -rf results/*
mkdir -p results

echo "=== gold (real text8 test, L=256) ==="
run_one sampler=gold model_id=gold exp_name="gold_L256_seed12345"

echo "=== M1 full CFM (steps=$STEPS) ==="
run_one sampler=full_cfm "sampler.steps=$STEPS" model_id=M1 \
  exp_name="m1_full_L256_s${STEPS}_seed12345"

echo "=== M2 block masked (B=$B, steps/blk=$STEPS) ==="
run_one sampler=bcfm_infer "sampler.block_size=$B" "sampler.steps_per_block=$STEPS" model_id=M2 \
  exp_name="m2_block_B${B}_s${STEPS}_L256_seed12345"

python -m eval.aggregate
echo "Done."
