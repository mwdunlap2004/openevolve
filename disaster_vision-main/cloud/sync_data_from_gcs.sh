#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BUCKET:-}" ]]; then
  echo "BUCKET must be set, for example: gs://your-disaster-vision-bucket" >&2
  exit 2
fi

DATA_ROOT=${DATA_ROOT:-"$HOME/data/xview2_jpeg"}
TRAIN_PREFIX=${TRAIN_PREFIX:-"$BUCKET/data/xview2_jpeg/tier1"}
VAL_PREFIX=${VAL_PREFIX:-"$BUCKET/data/xview2_jpeg/hold"}

mkdir -p "$DATA_ROOT"

gsutil -m rsync -r "$TRAIN_PREFIX" "$DATA_ROOT/tier1"
gsutil -m rsync -r "$VAL_PREFIX" "$DATA_ROOT/hold"

find "$DATA_ROOT" -maxdepth 3 -type d | sort

