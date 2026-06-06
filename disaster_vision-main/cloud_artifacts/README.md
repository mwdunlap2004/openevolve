# Disaster Vision Cloud Artifacts

Downloaded from the Google Cloud VM `disaster-vision-gpu` on June 6, 2026.

This directory contains GitHub-safe reproducibility artifacts from the GPU-backed OpenEvolve run:

- `logs/`: VM run logs for OpenEvolve and setup steps.
- `openevolve_output/`: OpenEvolve checkpoints, generated candidate programs, and metadata.
- `openevolve_gpu_trials/`: per-candidate 10-epoch GPU training trial configs, metrics, plots, and logs.

Excluded intentionally:

- raw xView2 data (`data/`, `data.zip`), because it is multi-GB and should not be committed to GitHub;
- model weights (`*.pt`, `*.pth`), because they are generated binary artifacts better suited for cloud storage or Git LFS;
- API keys and environment files.

Best GPU-backed candidate from the 60-iteration loop:

```text
combined_score: 0.4384530610
val_joint_score: 0.3907848591
val_dice: 0.5814576670
val_iou: 0.5814576670
val_cls_f1: 0.2001120512
best_epoch: 7
run_dir: openevolve_gpu_trials/oe_gpu_siamese_ordinal_baseline_1780710636_c9908c36
```

Best recipe:

```text
input_mode: siamese
encoder_mode: shared
fusion_strategy: concat_absdiff
fusion_stage: bottleneck_only
base_channels: 32
batch_size: 4
image_size: 256
lr: 0.0005
cls_weight: 2.0
damage_loss_mode: ce
damage_loss_alpha: 0.5
early_stopping_patience: 8
```
