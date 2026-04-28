#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
SKIP_OLLAMA_PULL="${SKIP_OLLAMA_PULL:-0}"
SKIP_SPACY_MODEL="${SKIP_SPACY_MODEL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_EXE}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${PROJECT_ROOT}/${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment python not found at ${VENV_PYTHON}" >&2
  exit 1
fi

echo "Upgrading pip"
"${VENV_PYTHON}" -m pip install --upgrade pip

echo "Installing requirements"
"${VENV_PYTHON}" -m pip install -r requirements.txt

if [[ "${SKIP_SPACY_MODEL}" != "1" ]]; then
  echo "Installing spaCy model: en_core_web_sm"
  "${VENV_PYTHON}" -m spacy download en_core_web_sm
fi

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example"
  cp .env.example .env
fi

"${VENV_PYTHON}" - <<'PY'
from pathlib import Path

env_path = Path(".env")
if not env_path.exists():
    raise SystemExit(0)

text = env_path.read_text(encoding="utf-8")
lines = text.splitlines()
updates = {
    "DEPLOYMENT_MODE": "local",
    "ENABLE_VERIFICATION": "true",
}
remove_keys = {"LLM_PROVIDER", "VECTOR_STORE"}

seen = set()
out = []
for line in lines:
    if any(line.startswith(f"{key}=") for key in remove_keys):
        continue
    replaced = False
    for key, value in updates.items():
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            seen.add(key)
            replaced = True
            break
    if not replaced:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

env_path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
PY

mkdir -p data/raw data/processed data/index data/eval

if [[ "${SKIP_OLLAMA_PULL}" != "1" ]]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "Pulling Ollama model: llama3.2:3b"
    ollama pull llama3.2:3b
else
    echo "Warning: ollama not found. Install Ollama and run: ollama pull llama3.2:3b" >&2
fi
fi

echo
echo "Local setup complete."
echo "Verification is enabled by default for the full claim analysis pipeline."
echo "Run: ./scripts/run_local.sh"
