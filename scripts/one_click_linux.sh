#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="config/experiment.json"
MODEL_NAME="llama3.1:8b"
SKIP_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--config <path>] [--model <name>] [--skip-run]"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] Repository root: $REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 is required but was not found. Install Python 3.10+ and re-run."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
fi

VENV_PYTHON=".venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment python not found at $VENV_PYTHON"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null

if [[ -f "requirements.txt" ]]; then
  if grep -Eq '^[[:space:]]*[^#[:space:]]' requirements.txt; then
    echo "[setup] Installing python dependencies..."
    "$VENV_PYTHON" -m pip install -r requirements.txt
  fi
fi

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "[setup] Created .env from .env.example"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[setup] Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! ollama list >/dev/null 2>&1; then
  echo "[setup] Starting Ollama server..."
  mkdir -p logs
  nohup ollama serve > logs/ollama-serve.log 2>&1 &
  sleep 3
fi

echo "[setup] Ensuring model '$MODEL_NAME' is available locally..."
ollama pull "$MODEL_NAME"

export EXPERIMENT_CONFIG_PATH="$CONFIG_PATH"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

if [[ "$SKIP_RUN" == "true" ]]; then
  echo "[done] Setup completed. Run manually with:"
  echo "       $VENV_PYTHON -m src.main --config $CONFIG_PATH"
  exit 0
fi

echo "[run] Starting experiment with config '$CONFIG_PATH'..."
"$VENV_PYTHON" -m src.main --config "$CONFIG_PATH"
