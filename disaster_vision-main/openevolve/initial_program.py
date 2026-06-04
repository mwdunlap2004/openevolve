#!/usr/bin/env python3
"""
Starting program for evolving the disaster-vision change-detection recipe.

OpenEvolve mutates the configuration returned here. The evaluator scores the
config against a surrogate objective derived from the strongest historical runs
in `jobs/leaderboard.csv`, so the task can run without the private dataset.
"""


# EVOLVE-BLOCK-START
def build_experiment_config():
    """
    Return a candidate training recipe for the disaster-vision project.

    The baseline starts from the simplest competitive recipe and leaves room
    for evolution toward the stronger shared-encoder Siamese variants.
    """

    return {
        "experiment_name": "after_only_baseline",
        "input_mode": "after_only",
        "encoder_mode": "shared",
        "fusion_strategy": "concat",
        "fusion_stage": "bottleneck_only",
        "base_channels": 32,
        "epochs": 50,
        "batch_size": 4,
        "image_size": 256,
        "lr": 1e-3,
        "cls_weight": 1.0,
        "damage_loss_mode": "ce",
        "damage_loss_alpha": 0.5,
        "selection_metric": "joint",
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 0.0,
    }


def run_experiment():
    return build_experiment_config()


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import json

    print(json.dumps(run_experiment(), indent=2))
