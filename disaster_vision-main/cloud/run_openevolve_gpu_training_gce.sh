#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$HOME/openevolve"}
ITERATIONS=${ITERATIONS:-60}
OE_TRAINING_EPOCHS=${OE_TRAINING_EPOCHS:-10}
OE_TRAINING_TIMEOUT_SECONDS=${OE_TRAINING_TIMEOUT_SECONDS:-7200}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set before running OpenEvolve." >&2
  exit 2
fi

cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-"$REPO_ROOT/.venv/bin/python"}

export OE_TRAINING_EPOCHS
export OE_TRAINING_TIMEOUT_SECONDS
export OE_TRAINING_RUNS_ROOT=${OE_TRAINING_RUNS_ROOT:-"$REPO_ROOT/disaster_vision-main/jobs/runs/openevolve_gpu_trials"}
export TRAIN_ROOT=${TRAIN_ROOT:-"$HOME/data/xview2_jpeg/tier1"}
export VAL_ROOT=${VAL_ROOT:-"$HOME/data/xview2_jpeg/hold"}

"$PYTHON_BIN" openevolve-run.py \
  disaster_vision-main/openevolve/initial_program.py \
  disaster_vision-main/openevolve/evaluator_gpu_training.py \
  --config disaster_vision-main/openevolve/config_gpu_training.yaml \
  --iterations "$ITERATIONS"
