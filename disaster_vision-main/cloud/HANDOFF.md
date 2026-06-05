# Disaster Vision / OpenEvolve Cloud Handoff

Last updated: June 5, 2026, 12:06 PM ET

This handoff is for continuing the `disaster_vision-main` work on the Google
Cloud GPU VM. The current focus is running OpenEvolve overnight to evolve better
training recipes, then using the best recipe for real disaster-vision model
training.

## Current Status

OpenEvolve is now working through OpenRouter with:

```text
model: qwen/qwen3.6-plus
api_base: https://openrouter.ai/api/v1
parallel_evaluations: 1
max_iterations: 900
early_stopping_patience: 900
checkpoint_interval: 10
```

The first successful run improved the surrogate score:

```text
baseline combined_score: 0.8684
best observed combined_score: 1.0372
best recipe family: siamese_bottleneck_concat_absdiff
```

The OpenRouter key works with `qwen/qwen3.6-plus`. Earlier free OpenRouter
models failed because of OpenRouter workspace privacy/guardrail policy.

## Google Cloud Project

```text
Project ID: project-01a04f95-aba2-4967-a5c
Account: upw4ys@virginia.edu
```

The old CPU VM was deleted. The active VM is:

```text
Name: disaster-vision-gpu
Zone: us-west1-a
Machine: g2-standard-8
GPU: NVIDIA L4, 23034 MiB
Boot disk: 250 GB
Image: Ubuntu accelerator image with NVIDIA driver 580
```

SSH:

```bash
gcloud compute ssh disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

Repo on VM:

```bash
~/openevolve
```

Project subdir:

```bash
~/openevolve/disaster_vision-main
```

Python env:

```bash
~/openevolve/.venv
```

## Live OpenEvolve Run

At handoff time, the original 60-iteration run is still active:

```text
wrapper PID: 18635
main PID: 18637
worker PID: 18646
```

An overnight supervisor is queued:

```text
supervisor PID: 18962
script: ~/openevolve/disaster_vision-main/logs/run_openevolve_overnight_qwen36plus.sh
```

The supervisor waits for the active 60-iteration run to finish, then resumes
from the latest checkpoint for 900 more iterations.

Check process status:

```bash
pgrep -af "openevolve-run|run_openevolve|overnight_qwen36plus"
```

Watch the active run:

```bash
tail -f "$(ls -t ~/openevolve/disaster_vision-main/logs/openevolve_qwen36plus_*.log | head -1)"
```

Watch the overnight supervisor:

```bash
tail -f ~/openevolve/disaster_vision-main/logs/openevolve_overnight_supervisor.out
```

When the overnight run starts, find its log:

```bash
ls -t ~/openevolve/disaster_vision-main/logs/openevolve_qwen36plus_overnight_*.log | head -1
```

Then tail it:

```bash
tail -f "$(ls -t ~/openevolve/disaster_vision-main/logs/openevolve_qwen36plus_overnight_*.log | head -1)"
```

## API Key Handling

Do not commit API keys.

The VM uses:

```bash
~/.disaster_vision_env
```

Expected content:

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
```

Load and verify without printing the key:

```bash
source ~/.disaster_vision_env
echo "${#OPENROUTER_API_KEY}"
```

If the length is `0`, reset it:

```bash
cat > ~/.disaster_vision_env
```

Paste:

```bash
export OPENROUTER_API_KEY='sk-or-v1-your-key-here'
```

Press `Ctrl+D`, then:

```bash
chmod 600 ~/.disaster_vision_env
source ~/.disaster_vision_env
echo "${#OPENROUTER_API_KEY}"
```

Direct API smoke test:

```bash
curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen3.6-plus","messages":[{"role":"user","content":"Reply OK only."}],"max_tokens":20}'
```

Success should return a normal chat completion. A `404` with
`guardrail restrictions and data policy` means the OpenRouter key/workspace is
blocking that model.

## Important Files

Local repo files:

```text
disaster_vision-main/openevolve/config.yaml
disaster_vision-main/openevolve/initial_program.py
disaster_vision-main/openevolve/evaluator.py
disaster_vision-main/cloud/run_openevolve_gce.sh
disaster_vision-main/cloud/run_training_gce.sh
disaster_vision-main/cloud/capture_process_artifacts.sh
disaster_vision-main/cloud/make_process_storyboard.py
```

VM logs:

```text
~/openevolve/disaster_vision-main/logs/openevolve_qwen36plus_20260605_160044.log
~/openevolve/disaster_vision-main/logs/openevolve_overnight_supervisor.out
~/openevolve/disaster_vision-main/logs/openevolve_overnight_wait_20260605_160525.log
```

VM checkpoints:

```text
~/openevolve/disaster_vision-main/openevolve/openevolve_output/checkpoints/checkpoint_10
~/openevolve/disaster_vision-main/openevolve/openevolve_output/checkpoints/checkpoint_20
```

More checkpoints should appear every 10 iterations.

## Extract Best OpenEvolve Result

After the run finishes, inspect logs for best programs:

```bash
cd ~/openevolve/disaster_vision-main
grep -R "New best program\\|🌟 New best solution\\|combined_score" logs/openevolve_qwen36plus*.log | tail -80
grep -R "candidate_config" openevolve/openevolve_output | tail -40
```

Find final checkpoints:

```bash
find openevolve/openevolve_output/checkpoints -maxdepth 1 -type d -name "checkpoint_*" | sort -V | tail
```

Use the best `candidate_config` as the recipe for real training.

## Dataset Status

Current VM dataset state:

```text
~/openevolve/disaster_vision-main/data.zip exists, about 6.8 GB
~/openevolve/disaster_vision-main/data is not extracted yet, about 20 KB
```

The previous extraction failed because `unzip` was not installed. To finish:

```bash
sudo apt-get update
sudo apt-get install -y unzip
cd ~/openevolve/disaster_vision-main
mkdir -p data
unzip -q data.zip -d data
du -sh data
find data -maxdepth 4 -type d | head -80
```

After extraction, identify the actual train/validation roots. Training expects
split folders shaped like:

```text
split_root/
  images_jpeg/
  labels/
```

## Real Training Run

After choosing a recipe and extracting data, run a smoke job first.

Example:

```bash
cd ~/openevolve
source .venv/bin/activate

TRAIN_ROOT=/path/to/train_split \
VAL_ROOT=/path/to/val_split \
EPOCHS=1 \
BATCH_SIZE=4 \
LR=3e-4 \
CLS_WEIGHT=2.0 \
DAMAGE_LOSS_MODE=ce \
INPUT_MODE=siamese \
FUSION_STRATEGY=concat_absdiff \
FUSION_STAGE=bottleneck_only \
ENCODER_MODE=shared \
bash disaster_vision-main/cloud/run_training_gce.sh
```

Then run the longer job with more epochs after the smoke test passes.

## Capture Evidence For Video / Presentation

The repo includes scripts for making a process storyboard from logs,
checkpoints, and model artifacts:

```bash
cd ~/openevolve
bash disaster_vision-main/cloud/capture_process_artifacts.sh
```

Expected outputs are under:

```text
disaster_vision-main/process_artifacts/
```

Use these for screenshots/video:

- OpenEvolve log snippets showing iterations and new best programs.
- Candidate configs from successful and failed programs.
- MAP-Elites/checkpoint artifacts.
- Training outputs once real training runs.

## Cost Control

The GPU VM is billable while running. Stop it when not actively running:

```bash
gcloud compute instances stop disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

Restart later:

```bash
gcloud compute instances start disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

## Known Issues

- OpenRouter free models were blocked by workspace privacy/guardrail policy.
- `qwen/qwen3.6-plus` works with the current OpenRouter key.
- Gemini API worked but hit free-tier quota quickly.
- OpenAI API would require separate paid API billing; ChatGPT Plus is separate.
- Big Pickle is an OpenCode Zen model and needs `OPENCODE_API_KEY`; it does not
  work with an OpenRouter key through the raw OpenEvolve API path.
- Some evolved candidates fail validation by inventing invalid values such as
  `input_mode: before_after` or `fusion_strategy: concat+absdiff`. That is normal
  search noise; the evaluator rejects those candidates with score `0.0`.

## Immediate Next Steps For Friend

1. SSH into `disaster-vision-gpu`.
2. Confirm OpenEvolve is still running or the overnight supervisor has started.
3. Let the overnight run finish unless there are repeated API errors.
4. Extract the best evolved `candidate_config`.
5. Install `unzip` and extract `data.zip`.
6. Locate train/validation split roots.
7. Run a one-epoch training smoke test using the best recipe.
8. Run the full training job.
9. Capture process artifacts for the final video/demo.
10. Stop the GPU VM when idle.
