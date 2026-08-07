#!/bin/bash
#SBATCH --job-name=bd3-h256-l3
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# Cluster-specific: uncomment / adjust for your site.
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --requeue

# Latency-matched specialization of run_tinystories_bd3lm.sbatch.
#
# The parent launcher remains the single implementation of the complete BD3
# workflow: environment setup, TinyStories preparation, strict same-shape MDLM
# compatibility checks, automatic MDLM train/resume, BD3 train/resume, and final
# evaluation.  This launcher changes only the model/config/run-name defaults:
#
#   block size        16
#   layers             3
#   hidden size      256
#   attention heads    4
#   MLP width        1024
#   parameters  28,792,914
#
# Submit from the BCFM Baselines repository:
#
#   sbatch slurm/run_tinystories_bd3lm_latency_h256_l3.sbatch
#
# All overrides documented by run_tinystories_bd3lm.sbatch continue to work,
# including SEED, BATCH_SIZE, GRAD_ACCUM, MAX_STEPS, VAL_EVERY, NUM_SAMPLES,
# EVAL_BATCH_SIZE, TINYSTORIES_DATA_DIR, source-data paths, AUTO_TRAIN_MDLM,
# BD3_PRETRAIN_CHECKPOINT, RUN_TIMESTAMP, and RUN_NAME.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$PROJECT_DIR/pyproject.toml" || ! -d "$PROJECT_DIR/bcfm_baselines" ]]; then
  if [[ -f "$PROJECT_DIR/BCFM Baselines/pyproject.toml" ]]; then
    PROJECT_DIR="$PROJECT_DIR/BCFM Baselines"
  else
    echo "ERROR: submit from the BCFM Baselines repository or set PROJECT_DIR." >&2
    exit 2
  fi
fi

export PROJECT_DIR
export BLOCK_SIZE=16
export CONFIG="${CONFIG:-configs/tinystories/bd3lm_b16_latency_h256_l3.yaml}"
export MDLM_CONFIG="${MDLM_CONFIG:-configs/tinystories/mdlm_latency_h256_l3.yaml}"
export MDLM_RUN="${MDLM_RUN:-mdlm_h256_l3_tinystories_seed${SEED:-12345}}"
export MDLM_COMPAT_RUN="${MDLM_COMPAT_RUN:-mdlm_h256_l3_tinystories_compat_seed${SEED:-12345}}"

# Keep the parent's timestamped-run convention while making the architecture
# explicit in the directory name. Reuse RUN_NAME or RUN_TIMESTAMP to resume.
if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  export RUN_TIMESTAMP
  export RUN_NAME="${RUN_TIMESTAMP}_bd3lm_b16_h256_l3_tinystories_seed${SEED:-12345}"
fi

exec bash "$PROJECT_DIR/slurm/run_tinystories_bd3lm.sbatch"
