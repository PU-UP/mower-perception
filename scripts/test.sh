#!/usr/bin/env bash
# Run from anywhere; use the repository's existing Python and Node environments.
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

echo 'Running Python tests'
"$PYTHON" -m pytest tests -q
echo 'Running frontend lint'
npm run lint
echo 'Checking TypeScript'
./node_modules/.bin/tsc --noEmit
echo 'Building production frontend'
npm run build
echo 'All checks passed. Live API and dataset evaluation commands are in README.md.'
