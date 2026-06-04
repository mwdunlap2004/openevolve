#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$HOME/openevolve"}
PROJECT_ROOT=${PROJECT_ROOT:-"$REPO_ROOT/disaster_vision-main"}
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/process_artifacts"}
OPENEVOLVE_OUTPUT=${OPENEVOLVE_OUTPUT:-"$PROJECT_ROOT/openevolve/openevolve_output"}
SEGFORMER_OUTPUTS=${SEGFORMER_OUTPUTS:-"$PROJECT_ROOT/SegFormer/outputs"}

cd "$REPO_ROOT"

args=(
  --openevolve-output "$OPENEVOLVE_OUTPUT"
  --segformer-outputs "$SEGFORMER_OUTPUTS"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${TRAINING_RUN:-}" ]]; then
  args+=(--training-run "$TRAINING_RUN")
fi

if [[ "${MAKE_VIDEO:-0}" == "1" ]]; then
  args+=(--make-video)
fi

python "$PROJECT_ROOT/cloud/make_process_storyboard.py" "${args[@]}"

