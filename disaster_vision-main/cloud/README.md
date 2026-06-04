# Google Cloud runbook

This runbook moves the existing `disaster_vision-main` Slurm workflow onto a
Google Cloud GPU VM. It covers two jobs:

- model training with `jobs/change_family_experiments.py`
- OpenEvolve looping with `openevolve-run.py` and the disaster-vision surrogate

The training script expects local filesystem paths with this structure:

```text
train_or_val_root/
  images_jpeg/
  labels/
```

Stage the dataset in Cloud Storage, then copy it to the VM local disk before
running training.

## 1. Choose project settings

Set these values locally before creating or using the VM:

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"
export ZONE="us-central1-a"
export BUCKET="gs://your-disaster-vision-bucket"
export VM_NAME="disaster-vision-gpu"
export MACHINE_TYPE="g2-standard-8"
export GPU_TYPE="nvidia-l4"
export GPU_COUNT="1"
```

The `g2-standard-8` plus one L4 GPU is a practical default for this repo. Use a
larger A2/G2 shape if you raise image size, batch size, or model width.

## 2. Create a GPU VM

From your local machine:

```bash
gcloud config set project "$PROJECT_ID"

gcloud compute instances create "$VM_NAME" \
  --zone "$ZONE" \
  --machine-type "$MACHINE_TYPE" \
  --accelerator "type=$GPU_TYPE,count=$GPU_COUNT" \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --boot-disk-size 250GB \
  --image-family pytorch-latest-gpu \
  --image-project deeplearning-platform-release \
  --metadata install-nvidia-driver=True
```

If GPU quota is unavailable in the chosen zone, switch zones or request quota
for the selected GPU type.

## 3. Upload data and code

Upload the xView2-style data once:

```bash
gsutil -m rsync -r /local/path/to/xview2_jpeg/tier1 "$BUCKET/data/xview2_jpeg/tier1"
gsutil -m rsync -r /local/path/to/xview2_jpeg/hold "$BUCKET/data/xview2_jpeg/hold"
```

Upload the repo snapshot you want to run:

```bash
gcloud compute scp --recurse . "$VM_NAME:~/openevolve" --zone "$ZONE"
```

## 4. Prepare the VM

SSH into the VM:

```bash
gcloud compute ssh "$VM_NAME" --zone "$ZONE"
```

Then run:

```bash
cd ~/openevolve
bash disaster_vision-main/cloud/setup_gce.sh
```

Copy the dataset from Cloud Storage to local disk:

```bash
export BUCKET="gs://your-disaster-vision-bucket"
bash disaster_vision-main/cloud/sync_data_from_gcs.sh
```

This creates:

```text
~/data/xview2_jpeg/tier1
~/data/xview2_jpeg/hold
```

## 5. Run one model training job

Use the strongest current recipe defaults from the project history:

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

Outputs are written to:

```text
disaster_vision-main/jobs/runs/<run_id>/
```

To persist results back to Cloud Storage:

```bash
export BUCKET="gs://your-disaster-vision-bucket"
gsutil -m rsync -r disaster_vision-main/jobs/runs "$BUCKET/runs"
```

## 6. Run the OpenEvolve loop

The current OpenEvolve task uses the surrogate evaluator in
`disaster_vision-main/openevolve/evaluator.py`, so it does not need the full
dataset. It does require an LLM API key.

```bash
cd ~/openevolve
source .venv/bin/activate
export OPENROUTER_API_KEY="..."

ITERATIONS=60 bash disaster_vision-main/cloud/run_openevolve_gce.sh
```

Checkpoints are written under the configured OpenEvolve output directory. After
the loop finishes, inspect the best evolved program and copy its returned config
into `run_training_gce.sh` environment variables for a real cloud training run.

## 7. Stop or delete the VM

Stop the VM when you are not actively using the GPU:

```bash
gcloud compute instances stop "$VM_NAME" --zone "$ZONE"
```

Delete it when the run is complete and outputs are synced:

```bash
gcloud compute instances delete "$VM_NAME" --zone "$ZONE"
```

