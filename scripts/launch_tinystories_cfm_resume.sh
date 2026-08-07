#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/coder/project/.CFM-DIFF-TEXT/cfm-text-distillation-smiles-2026"
PYTHON_BIN="/home/coder/.conda/envs/semicat/bin/python"
RUN_DIR="$PROJECT_DIR/logs/train/tinystories_cfm_resume75k_to200k_nfe32_20260731"
PID_FILE="$RUN_DIR/train.pid"
CONSOLE_LOG="$RUN_DIR/console.log"

mkdir -p "$RUN_DIR" "$PROJECT_DIR/artifacts/cache/triton"

if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID="$(<"$PID_FILE")"
    if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Training is already running: pid=$EXISTING_PID run_dir=$RUN_DIR"
        exit 0
    fi
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PROJECT_DIR/artifacts/cache/triton"
export OMP_NUM_THREADS=8

COMMAND=(
    "$PYTHON_BIN" -m semicat.train
    experiment=cfm_tinystories_resume_nfe32
    logger=csv
    "hydra.run.dir=$RUN_DIR"
)

nohup setsid "${COMMAND[@]}" >"$CONSOLE_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" >"$PID_FILE"
{
    printf 'launched_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'pid=%s\n' "$TRAIN_PID"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'checkpoint=%s\n' "$PROJECT_DIR/baseline/tinystories/last (2).ckpt"
    printf 'target_global_step=200000\n'
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} >"$RUN_DIR/launch_manifest.txt"

echo "Launched TinyStories CFM training: pid=$TRAIN_PID"
echo "Run directory: $RUN_DIR"
echo "Console log: $CONSOLE_LOG"
