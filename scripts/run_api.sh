#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PYTHON="${PROJECT_ROOT}/${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}. Run ./scripts/setup_local.sh first." >&2
  exit 1
fi

export DEPLOYMENT_MODE=local
export API_HOST="${API_HOST:-127.0.0.1}"
export API_PORT="${API_PORT:-8000}"
export ENABLE_VERIFICATION="${ENABLE_VERIFICATION:-true}"
export APP_LOG_LEVEL="${APP_LOG_LEVEL:-INFO}"

if [[ ! -f .env ]]; then
  echo "Warning: .env not found. Run ./scripts/setup_local.sh first to create a local default .env." >&2
fi

echo "Starting FastAPI in local mode..."
echo "DEPLOYMENT_MODE=${DEPLOYMENT_MODE} API_HOST=${API_HOST} API_PORT=${API_PORT} ENABLE_VERIFICATION=${ENABLE_VERIFICATION} APP_LOG_LEVEL=${APP_LOG_LEVEL}"

exec "${VENV_PYTHON}" -m uvicorn src.api.main:app --host "${API_HOST}" --port "${API_PORT}" --log-level "${APP_LOG_LEVEL,,}"
