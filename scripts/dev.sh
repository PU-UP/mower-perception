#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

API_PORT="${API_PORT:-43131}"
WEB_PORT="${WEB_PORT:-43129}"

"$PYTHON" -m uvicorn mowerseg.server:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

npx --yes wait-on "http-get://127.0.0.1:${API_PORT}/api/health" --timeout 180000
npm run dev -- --hostname 127.0.0.1 --port "$WEB_PORT"
