#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$HOME/openevolve"}
VENV_DIR=${VENV_DIR:-"$REPO_ROOT/.venv"}

cd "$REPO_ROOT"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r disaster_vision-main/requirements.txt
python -m pip install -e .

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

