#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_DIR="${LOG_DIR:-logs}"
LIMIT="${LIMIT:-0}"
VERBOSE_DOWNLOAD="${VERBOSE_DOWNLOAD:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_PYTHON="${PROJECT_ROOT}/${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}. Run ./scripts/setup_local.sh first." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d-%H%M%S)"
log_path="${LOG_DIR}/weekly-corpus-update-${timestamp}.log"

cmd=( "${VENV_PYTHON}" "scripts/download_corpus.py" "--update" )
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=( "--output-dir" "${OUTPUT_DIR}" )
fi
if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && [[ "${LIMIT}" -gt 0 ]]; then
  cmd+=( "--limit" "${LIMIT}" )
fi
if [[ "${VERBOSE_DOWNLOAD}" == "1" ]]; then
  cmd+=( "--verbose" )
fi

echo "Starting weekly corpus update..."
echo "Project root: ${PROJECT_ROOT}"
echo "Log: ${log_path}"
echo "Command: ${cmd[*]}"

{
  printf '[%s] START weekly corpus update\n' "$(date -Iseconds)"
  printf '[%s] Command: %s\n' "$(date -Iseconds)" "${cmd[*]}"
} > "${log_path}"

set +e
"${cmd[@]}" 2>&1 | tee -a "${log_path}"
exit_code=${PIPESTATUS[0]}
set -e

printf '[%s] EXIT code=%s\n' "$(date -Iseconds)" "${exit_code}" >> "${log_path}"

if [[ "${exit_code}" -ne 0 ]]; then
  echo "Weekly corpus update failed with exit code ${exit_code}. See log: ${log_path}" >&2
  exit "${exit_code}"
fi

echo "Weekly corpus update completed successfully."
echo "Log saved to: ${log_path}"
