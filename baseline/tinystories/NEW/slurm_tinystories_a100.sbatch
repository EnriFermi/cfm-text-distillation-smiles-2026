#!/bin/bash
#SBATCH --job-name=semicat-ts
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err    
#SBATCH --nodelist=michman498212ab-44e5-4eb7-99c9-b042df4caf42
#SBATCH --cpus-per-task=4          
#SBATCH --mem=32G                  
#SBATCH --partition=cpu
#SBATCH --gres=gpu:GA100:1    


set -euo pipefail

# Submit from the repository root:
#   sbatch slurm_tinystories_a100.sbatch
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$PROJECT_DIR/.project-root" ]]; then
    echo "ERROR: $PROJECT_DIR is not the semicat repository root (.project-root not found)." >&2
    echo "Run: cd /path/to/semicat && sbatch slurm_tinystories_a100.sbatch" >&2
    exit 2
fi
cd "$PROJECT_DIR"

ENV_DIR="${ENV_DIR:-$PROJECT_DIR/.venv}"

python_is_compatible() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 14) else 1)' \
        >/dev/null 2>&1
}

# The codebase supports Python >=3.9. Prefer a versioned binary
# because many clusters still expose Python 3.8/3.9 as plain `python3`.
if [[ -n "${PYTHON_BOOTSTRAP:-}" ]]; then
    PYTHON_CANDIDATES=("$PYTHON_BOOTSTRAP")
else
    PYTHON_CANDIDATES=(python3.11 python3.10 python3.12 python3.9 python3 python)
fi
PYTHON_BOOTSTRAP=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && python_is_compatible "$candidate"; then
        PYTHON_BOOTSTRAP="$(command -v "$candidate")"
        break
    fi
done
if [[ -z "$PYTHON_BOOTSTRAP" ]]; then
    echo "ERROR: Python 3.9--3.12 was not found." >&2
    echo "Load it before sbatch (for example: module load python/3.11)," >&2
    echo "or submit with --export=ALL,PYTHON_BOOTSTRAP=/path/to/python3.11" >&2
    exit 3
fi
echo "Bootstrap:  $PYTHON_BOOTSTRAP ($("$PYTHON_BOOTSTRAP" --version 2>&1))"

if [[ -x "$ENV_DIR/bin/python" ]] && ! python_is_compatible "$ENV_DIR/bin/python"; then
    OLD_ENV="${ENV_DIR}.incompatible-${SLURM_JOB_ID:-local}"
    echo "Moving incompatible environment to $OLD_ENV"
    mv "$ENV_DIR" "$OLD_ENV"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Creating Python environment at $ENV_DIR"
    "$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
    "$ENV_DIR/bin/python" -m pip install --upgrade pip
fi
PYTHON="$ENV_DIR/bin/python"

# This is intentionally also run for an existing environment: an interrupted
# first job may leave a valid venv with only part of the requirements installed.
"$PYTHON" -m pip install -r "$PROJECT_DIR/requirements-a100.txt"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_DIR/data/tinystories/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Reusing RUN_NAME resumes from this run's last.ckpt. Change RUN_NAME to start
# an independent experiment.
RUN_NAME="${RUN_NAME:-tinystories_a100}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/logs/tinystories/$RUN_NAME}"
LAST_CKPT="$RUN_DIR/checkpoints/last.ckpt"
mkdir -p "$RUN_DIR"

echo "Project:    $PROJECT_DIR"
echo "Python:     $PYTHON"
echo "Run:        $RUN_DIR"
echo "SLURM job:  ${SLURM_JOB_ID:-local}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

ARGS=(
    experiment=tinystories_dit
    trainer.devices=1
    "data.batch_size=${BATCH_SIZE:-32}"
    "data.num_workers=${DATA_NUM_WORKERS:-2}"
    "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES:-4}"
    "trainer.max_steps=${MAX_STEPS:-100000}"
    "trainer.val_check_interval=${VAL_EVERY_N_BATCHES:-4000}"
    "hydra.run.dir=$RUN_DIR"
)

if [[ -f "$LAST_CKPT" ]]; then
    echo "Resuming from $LAST_CKPT"
    ARGS+=("ckpt_path=$LAST_CKPT")
else
    echo "No checkpoint found; starting a new run"
fi

# srun gives Slurm correct process/signal tracking. After a timeout or preemption,
# submit this same file again and training will load last.ckpt automatically.
exec srun --unbuffered "$PYTHON" -m semicat.train "${ARGS[@]}"
