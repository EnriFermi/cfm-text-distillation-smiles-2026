#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${PROJECT_ROOT}/baseline/s_baseline.ckpt"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[BCFM] baseline checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
echo "[BCFM] stage=launch label=bcfm-finetune-text8"
echo "[BCFM] project_root=${PROJECT_ROOT}"
echo "[BCFM] checkpoint=${CHECKPOINT}"
echo "[BCFM] python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[BCFM] overrides=$*"

exec "${PYTHON_BIN}" -m semicat.train \
  experiment=bcfm_finetune_text8 \
  trainer=gpu \
  logger=comet \
  "$@"
