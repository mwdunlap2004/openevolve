#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/openevolve"
source "$HOME/.disaster_vision_env"
wait_log="disaster_vision-main/logs/openevolve_overnight_wait_$(date +%Y%m%d_%H%M%S).log"
echo "overnight supervisor started at $(date -Is)" | tee -a "$wait_log"
echo "waiting for existing OpenEvolve run to finish" | tee -a "$wait_log"
while pgrep -f "openevolve-run.py disaster_vision-main/openevolve" >/dev/null; do
  pgrep -af "openevolve-run|run_openevolve" | tee -a "$wait_log" || true
  sleep 60
done
latest_checkpoint=$(find disaster_vision-main/openevolve/openevolve_output/checkpoints -maxdepth 1 -type d -name "checkpoint_*" 2>/dev/null | sort -V | tail -1 || true)
run_log="disaster_vision-main/logs/openevolve_qwen36plus_overnight_$(date +%Y%m%d_%H%M%S).log"
echo "resuming at $(date -Is) from checkpoint: ${latest_checkpoint:-none}" | tee -a "$wait_log"
if [[ -n "${latest_checkpoint:-}" ]]; then
  .venv/bin/python openevolve-run.py \
    disaster_vision-main/openevolve/initial_program.py \
    disaster_vision-main/openevolve/evaluator.py \
    --config disaster_vision-main/openevolve/config.yaml \
    --checkpoint "$latest_checkpoint" \
    --iterations 900 \
    > "$run_log" 2>&1
else
  .venv/bin/python openevolve-run.py \
    disaster_vision-main/openevolve/initial_program.py \
    disaster_vision-main/openevolve/evaluator.py \
    --config disaster_vision-main/openevolve/config.yaml \
    --iterations 900 \
    > "$run_log" 2>&1
fi
echo "overnight run finished at $(date -Is), log=$run_log" | tee -a "$wait_log"
