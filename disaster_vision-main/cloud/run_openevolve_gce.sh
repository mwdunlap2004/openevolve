#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$HOME/openevolve"}
ITERATIONS=${ITERATIONS:-60}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set before running OpenEvolve." >&2
  exit 2
fi

cd "$REPO_ROOT"

python openevolve-run.py \
  disaster_vision-main/openevolve/initial_program.py \
  disaster_vision-main/openevolve/evaluator.py \
  --config disaster_vision-main/openevolve/config.yaml \
  --iterations "$ITERATIONS"

