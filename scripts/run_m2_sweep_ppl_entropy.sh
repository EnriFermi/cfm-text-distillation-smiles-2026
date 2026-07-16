#!/usr/bin/env bash
# M2 sweep: B ∈ {16,32} × steps_per_block ∈ {1,4,16}
# Also M1 steps ∈ {1,4,16} for a fair PPL-vs-NFE reference.
# Keeps existing results/ (does not wipe).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PROJECT_ROOT="$PWD"

CKPT=/root/checkpoints/last.ckpt
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
)

run_one() {
  echo ">>> $*"
  python -m eval.run_eval "$@" "${COMMON[@]}"
}

mkdir -p results

# gold reference (skip if present)
if [[ ! -f results/gold_L256_seed12345/metrics.json ]]; then
  run_one sampler=gold model_id=gold exp_name="gold_L256_seed12345" report_block_size=16
fi

echo "=== M1 full CFM steps sweep ==="
for s in 1 4 16; do
  exp="m1_full_L256_s${s}_seed12345"
  if [[ -f "results/${exp}/metrics.json" ]]; then
    echo "skip $exp (exists)"
    continue
  fi
  run_one sampler=full_cfm "sampler.steps=$s" model_id=M1 \
    exp_name="$exp" report_block_size=16
done

echo "=== M2 B × steps/block sweep ==="
for B in 16 32; do
  for s in 1 4 16; do
    exp="m2_block_B${B}_s${s}_L256_seed12345"
    if [[ -f "results/${exp}/metrics.json" ]]; then
      echo "skip $exp (exists)"
      continue
    fi
    run_one sampler=bcfm_infer "sampler.block_size=$B" "sampler.steps_per_block=$s" \
      model_id=M2 exp_name="$exp" report_block_size="$B"
  done
done

python -m eval.aggregate
echo "Done."
