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

if [[ ! -f .env ]]; then
  echo "Warning: .env not found. Run ./scripts/setup_local.sh first to create a local default .env." >&2
fi

echo "Starting Streamlit in local mode..."
echo "DEPLOYMENT_MODE=${DEPLOYMENT_MODE} LLM_MODEL=${LLM_MODEL:-} OLLAMA_HOST=${OLLAMA_HOST:-}"

exec "${VENV_PYTHON}" -m streamlit run src/app.py
