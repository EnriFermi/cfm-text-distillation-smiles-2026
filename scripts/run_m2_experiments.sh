#!/usr/bin/env bash
# M2 inference-time experiments: proper BD3-LM masked BCFM.
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
  seed=12345
  model_id=M2
)

run_one() {
  echo ">>> $*"
  python -m eval.run_eval "$@" "${COMMON[@]}"
}

echo "=== M2 B=16, steps/block sweep ==="
for s in 1 2 4; do
  run_one sampler=bcfm_infer sampler.block_size=16 "sampler.steps_per_block=$s"
done

echo "=== M2 block-size sweep (1 step/block) ==="
for b in 4 8 16 32; do
  run_one sampler=bcfm_infer "sampler.block_size=$b" sampler.steps_per_block=1
done

echo "=== aggregate ==="
python -m eval.aggregate
echo "Done."
