#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$HOME/openevolve"}
PROJECT_ROOT=${PROJECT_ROOT:-"$REPO_ROOT/disaster_vision-main"}
JOBS_ROOT="$PROJECT_ROOT/jobs"
RUNS_ROOT=${RUNS_ROOT:-"$JOBS_ROOT/runs"}
SCRIPT_PATH="$JOBS_ROOT/change_family_experiments.py"

TRAIN_ROOT=${TRAIN_ROOT:-"$HOME/data/xview2_jpeg/tier1"}
VAL_ROOT=${VAL_ROOT:-"$HOME/data/xview2_jpeg/hold"}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-siamese_shared_concat_absdiff_bottleneck_cloud}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
IMAGE_SIZE=${IMAGE_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-3e-4}
BASE_CHANNELS=${BASE_CHANNELS:-32}
CLS_WEIGHT=${CLS_WEIGHT:-2.0}
DAMAGE_LOSS_MODE=${DAMAGE_LOSS_MODE:-ordinal}
DAMAGE_LOSS_ALPHA=${DAMAGE_LOSS_ALPHA:-0.5}
DEBUG_SAMPLES=${DEBUG_SAMPLES:-0}
SELECTION_METRIC=${SELECTION_METRIC:-joint}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-12}
EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.0}

INPUT_MODE=${INPUT_MODE:-siamese}
ENCODER_MODE=${ENCODER_MODE:-shared}
FUSION_STRATEGY=${FUSION_STRATEGY:-concat_absdiff}
FUSION_STAGE=${FUSION_STAGE:-bottleneck_only}

RUN_ID="${EXPERIMENT_NAME}_gce_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUNS_ROOT/$RUN_ID"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

{
  echo "run_id=$RUN_ID"
  echo "hostname=$(hostname)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$REPO_ROOT"
  echo "train_root=$TRAIN_ROOT"
  echo "val_root=$VAL_ROOT"
  echo "experiment_name=$EXPERIMENT_NAME"
} > "$RUN_DIR/job_info.txt"

python --version | tee "$RUN_DIR/python-version.txt"
nvidia-smi | tee "$RUN_DIR/nvidia-smi.txt" || true

python "$SCRIPT_PATH" \
  --train_root "$TRAIN_ROOT" \
  --val_root "$VAL_ROOT" \
  --save_dir "$RUNS_ROOT" \
  --run_name "$RUN_ID" \
  --exp_name "$EXPERIMENT_NAME" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --image_size "$IMAGE_SIZE" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --base_channels "$BASE_CHANNELS" \
  --cls_weight "$CLS_WEIGHT" \
  --damage_loss_mode "$DAMAGE_LOSS_MODE" \
  --damage_loss_alpha "$DAMAGE_LOSS_ALPHA" \
  --debug_samples "$DEBUG_SAMPLES" \
  --selection_metric "$SELECTION_METRIC" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA" \
  --input_mode "$INPUT_MODE" \
  --encoder_mode "$ENCODER_MODE" \
  --fusion_strategy "$FUSION_STRATEGY" \
  --fusion_stage "$FUSION_STAGE" \
  2>&1 | tee "$RUN_DIR/train.log"

