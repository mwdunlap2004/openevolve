# Disaster Vision Google Cloud Handoff

Last updated: June 4, 2026

This document captures the current Google Cloud state and the exact next steps
for continuing the `disaster_vision-main` OpenEvolve and model-training work.

## Goal

Run two workflows in Google Cloud:

1. Run the OpenEvolve loop to evolve disaster-vision training recipes.
2. Run real model training for selected recipes on a GPU VM.

OpenEvolve currently uses a surrogate evaluator, so it can run on CPU. Real
model training should run on the GPU VM.

## Project And Account

Google Cloud account:

```bash
upw4ys@virginia.edu
```

Project:

```bash
project-01a04f95-aba2-4967-a5c
```

Compute Engine API is enabled. GPU quota was approved. A working US L4 GPU zone
was found by actively probing VM creation from the CLI.

## Running VMs

There are currently two running VMs.

```text
NAME                   ZONE           MACHINE_TYPE   EXTERNAL_IP
disaster-vision-setup  us-central1-a  e2-standard-8  35.193.173.61
disaster-vision-gpu    us-west1-a     g2-standard-8  35.233.213.207
```

The GPU VM is the important one for training:

```text
GPU VM: disaster-vision-gpu
Zone: us-west1-a
Machine: g2-standard-8
GPU: 1 NVIDIA L4
GPU memory: 23034 MiB
Driver: 580.159.03
CUDA runtime shown by nvidia-smi: 13.0
Root disk: 243G total, about 240G free before Python dependency setup
```

Verified on the GPU VM:

```bash
nvidia-smi
```

Output showed one `NVIDIA L4`.

## SSH Commands

GPU VM:

```bash
gcloud compute ssh disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

CPU setup VM:

```bash
gcloud compute ssh disaster-vision-setup \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-central1-a
```

## Current Repo State On VMs

Local repository root:

```bash
/Users/michaeldunlap/Desktop/Recents/UVA_MSDS_Work/big_data/openevolve
```

The repo archive was uploaded and unpacked on the GPU VM:

```bash
~/openevolve
```

The GPU VM already has the source tree, including:

```bash
~/openevolve/disaster_vision-main/cloud/README.md
~/openevolve/disaster_vision-main/cloud/HANDOFF.md
~/openevolve/disaster_vision-main/cloud/setup_gce.sh
~/openevolve/disaster_vision-main/cloud/sync_data_from_gcs.sh
~/openevolve/disaster_vision-main/cloud/run_openevolve_gce.sh
~/openevolve/disaster_vision-main/cloud/run_training_gce.sh
~/openevolve/disaster_vision-main/cloud/capture_process_artifacts.sh
~/openevolve/disaster_vision-main/cloud/make_process_storyboard.py
```

The GPU VM has system Python tooling installed:

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip
```

Next step on the GPU VM is to run the project setup script:

```bash
cd ~/openevolve
bash disaster_vision-main/cloud/setup_gce.sh
```

That script creates `~/openevolve/.venv`, installs Python dependencies, installs
OpenEvolve editable, and checks PyTorch/CUDA visibility.

The CPU setup VM already completed this setup and can run OpenEvolve. On that
CPU VM, PyTorch installed successfully but `cuda_available` is false, which is
expected.

## How The GPU VM Was Found

Initial GPU creation attempts failed due to `ZONE_RESOURCE_POOL_EXHAUSTED`.
An active CLI probe then found capacity in:

```text
zone=us-west1-a
machine=g2-standard-8
gpu=nvidia-l4
instance=disaster-vision-gpu
```

Successful creation command:

```bash
gcloud compute instances create disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --boot-disk-size 250GB \
  --image-family ubuntu-accelerator-2204-amd64-with-nvidia-580 \
  --image-project ubuntu-os-accelerator-images \
  --scopes cloud-platform
```

Earlier failed zones included `us-central1-a`, `us-central1-b`,
`us-central1-c`, and others. The current working zone is `us-west1-a`.

## Immediate Next Steps

1. SSH into the GPU VM.
2. Run the setup script on the GPU VM.
3. Confirm PyTorch sees CUDA.
4. Sync the dataset from Cloud Storage.
5. Run a one-epoch smoke training job.
6. Run the OpenEvolve loop if not already done.
7. Train the best evolved recipe.
8. Capture process artifacts for the final video/demo.
9. Stop idle VMs to control cost.

## Step 1: Finish GPU VM Setup

```bash
gcloud compute ssh disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

Then on the VM:

```bash
cd ~/openevolve
bash disaster_vision-main/cloud/setup_gce.sh
```

After setup, verify:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
```

Expected result on the GPU VM:

```text
cuda_available: True
cuda_device: NVIDIA L4
```

## Step 2: Dataset Sync

The training script expects this local structure:

```text
split_root/
  images_jpeg/
  labels/
```

Once the dataset is staged in Cloud Storage, run on the GPU VM:

```bash
cd ~/openevolve
source .venv/bin/activate
export BUCKET="gs://your-disaster-vision-bucket"
bash disaster_vision-main/cloud/sync_data_from_gcs.sh
```

Default local paths:

```bash
~/data/xview2_jpeg/tier1
~/data/xview2_jpeg/hold
```

If the bucket uses different paths, override:

```bash
TRAIN_PREFIX="$BUCKET/path/to/tier1" \
VAL_PREFIX="$BUCKET/path/to/hold" \
bash disaster_vision-main/cloud/sync_data_from_gcs.sh
```

## Step 3: Smoke Test Training

Run this on the GPU VM after data sync:

```bash
cd ~/openevolve
source .venv/bin/activate

EPOCHS=1 \
DEBUG_SAMPLES=2 \
BATCH_SIZE=2 \
bash disaster_vision-main/cloud/run_training_gce.sh
```

This should create a run under:

```bash
disaster_vision-main/jobs/runs/
```

Check:

```bash
ls -lah disaster_vision-main/jobs/runs
```

## Step 4: Full Training Recipe

After the smoke test works:

```bash
cd ~/openevolve
source .venv/bin/activate

EXPERIMENT_NAME=siamese_shared_concat_absdiff_bottleneck_cloud \
INPUT_MODE=siamese \
ENCODER_MODE=shared \
FUSION_STRATEGY=concat_absdiff \
FUSION_STAGE=bottleneck_only \
EPOCHS=100 \
BATCH_SIZE=8 \
LR=3e-4 \
CLS_WEIGHT=2.0 \
DAMAGE_LOSS_MODE=ordinal \
DAMAGE_LOSS_ALPHA=0.5 \
bash disaster_vision-main/cloud/run_training_gce.sh
```

If CUDA memory is tight, reduce:

```bash
BATCH_SIZE=4
```

If still tight:

```bash
BATCH_SIZE=2
IMAGE_SIZE=224
BASE_CHANNELS=16
```

## Step 5: Run OpenEvolve

OpenEvolve can run on either VM. Use the CPU setup VM to save GPU cost, or run
it on the GPU VM if that is simpler.

On either VM:

```bash
cd ~/openevolve
source .venv/bin/activate
export OPENROUTER_API_KEY="..."
ITERATIONS=60 bash disaster_vision-main/cloud/run_openevolve_gce.sh
```

Do not commit API keys. Use environment variables only.

OpenEvolve task files:

```bash
disaster_vision-main/openevolve/initial_program.py
disaster_vision-main/openevolve/evaluator.py
disaster_vision-main/openevolve/config.yaml
```

## Optional: OpenCode With Gemini

If using OpenCode CLI with Gemini:

```bash
export GEMINI_API_KEY="..."
opencode
```

Optional VM-local env file:

```bash
printf 'export GEMINI_API_KEY="your-key"\n' >> ~/.disaster_vision_env
printf 'source ~/.disaster_vision_env\n' >> ~/.bashrc
```

Do not put `GEMINI_API_KEY` or `OPENROUTER_API_KEY` in the repo.

## Step 6: Capture Process Artifacts

After OpenEvolve and at least one training run:

```bash
cd ~/openevolve
source .venv/bin/activate
bash disaster_vision-main/cloud/capture_process_artifacts.sh
```

Expected output:

```bash
disaster_vision-main/process_artifacts/
```

Contents:

```text
storyboard.md
manifest.csv
summary.json
frames/
evidence/
```

If `ffmpeg` is installed:

```bash
MAKE_VIDEO=1 bash disaster_vision-main/cloud/capture_process_artifacts.sh
```

## Sync Outputs Back To Cloud Storage

```bash
export BUCKET="gs://your-disaster-vision-bucket"
gsutil -m rsync -r disaster_vision-main/jobs/runs "$BUCKET/runs"
gsutil -m rsync -r disaster_vision-main/process_artifacts "$BUCKET/process_artifacts"
```

## Cost Controls

Both VMs are currently running. Stop anything idle.

Stop the CPU setup VM:

```bash
gcloud compute instances stop disaster-vision-setup \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-central1-a
```

Stop the GPU VM:

```bash
gcloud compute instances stop disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

Start the GPU VM:

```bash
gcloud compute instances start disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

Delete the GPU VM when done:

```bash
gcloud compute instances delete disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a
```

Before deleting, sync outputs back to Cloud Storage or local disk.

## Quick Status Commands

List VMs:

```bash
gcloud compute instances list \
  --project project-01a04f95-aba2-4967-a5c
```

Check GPU:

```bash
gcloud compute ssh disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a \
  --command 'nvidia-smi'
```

Check disk:

```bash
gcloud compute ssh disaster-vision-gpu \
  --project project-01a04f95-aba2-4967-a5c \
  --zone us-west1-a \
  --command 'df -h /'
```

