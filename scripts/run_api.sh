#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PYTHON="${PROJECT_ROOT}/${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  FALLBACK_PYTHON="${PROJECT_ROOT}/../${VENV_DIR}/bin/python"
  if [[ -x "${FALLBACK_PYTHON}" ]]; then
    VENV_PYTHON="${FALLBACK_PYTHON}"
  else
    echo "Virtual environment not found at ${VENV_DIR} or ../${VENV_DIR}. Run ./scripts/setup_local.sh first." >&2
    exit 1
  fi
fi

export DEPLOYMENT_MODE=local

echo "Starting FastAPI backend..."
echo "API at http://127.0.0.1:8000"

exec "${VENV_PYTHON}" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
