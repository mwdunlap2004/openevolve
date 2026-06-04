#!/usr/bin/env python3
"""
Surrogate evaluator for the disaster-vision OpenEvolve task.

The repository does not include the private xView2 dataset, so this evaluator
scores candidate recipes from the configuration they return. The score is
calibrated from the project history in `jobs/leaderboard.csv`:

- shared encoders are preferred over separate encoders
- bottleneck fusion is slightly stronger than multiscale for the joint score
- concat+absolute-difference fusion has been the best-performing family
- `cls_weight` around 2.0 and `lr` around 3e-4 are strong defaults

The evaluator returns `combined_score` so OpenEvolve can rank candidates
directly. If you later have access to the full dataset, this file can be
replaced with a real training loop that calls `jobs/change_family_experiments.py`.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_PATH = ROOT / "jobs" / "leaderboard.csv"

ALLOWED_VALUES = {
    "input_mode": {"after_only", "siamese"},
    "encoder_mode": {"shared", "separate"},
    "fusion_strategy": {"concat", "absdiff", "concat_absdiff"},
    "fusion_stage": {"bottleneck_only", "multiscale"},
    "damage_loss_mode": {"ce", "ordinal", "hierarchical"},
    "selection_metric": {"joint", "dice", "cls_f1", "cls_acc"},
}

REQUIRED_KEYS = [
    "experiment_name",
    "input_mode",
    "encoder_mode",
    "fusion_strategy",
    "fusion_stage",
    "base_channels",
    "epochs",
    "batch_size",
    "image_size",
    "lr",
    "cls_weight",
    "damage_loss_mode",
    "damage_loss_alpha",
    "selection_metric",
    "early_stopping_patience",
    "early_stopping_min_delta",
]


def evaluate(program_path: str) -> Dict[str, Any]:
    start_time = time.time()
    try:
        config = load_candidate_config(program_path)
        validation_errors = validate_config(config)
        if validation_errors:
            return error_result(
                validation_errors,
                start_time,
                config=config,
                status="VALIDATION_ERROR",
            )

        score, breakdown = score_config(config)
        elapsed = time.time() - start_time

        artifacts = {
            "candidate_config": json.dumps(config, indent=2, sort_keys=True),
            "score_breakdown": json.dumps(breakdown, indent=2, sort_keys=True),
            "historical_context": (
                "Best observed family: shared Siamese encoder with bottleneck "
                "fusion and concat/absdiff fusion. The after-only baseline is "
                "competitive, but the leaderboard favors richer change fusion."
            ),
            "leaderboard_path": str(LEADERBOARD_PATH),
            "leaderboard_exists": str(LEADERBOARD_PATH.exists()),
            "eval_time": f"{elapsed:.3f}s",
        }

        return {
            "combined_score": float(score),
            "architectural_score": float(breakdown["architectural_score"]),
            "training_score": float(breakdown["training_score"]),
            "regularization_score": float(breakdown["regularization_score"]),
            "stability_score": float(breakdown["stability_score"]),
            "eval_time": elapsed,
            "artifacts": artifacts,
        }
    except Exception as exc:
        return error_result(
            [f"{type(exc).__name__}: {exc}"],
            start_time,
            extra_artifacts={"traceback": traceback.format_exc()},
        )


def load_candidate_config(program_path: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program at {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("run_experiment", "build_experiment_config", "configure_experiment"):
        fn = getattr(module, name, None)
        if callable(fn):
            config = fn()
            if isinstance(config, dict):
                return config
            raise TypeError(f"{name}() must return a dict, got {type(config).__name__}")

    raise AttributeError(
        "Program must define run_experiment(), build_experiment_config(), "
        "or configure_experiment()"
    )


def validate_config(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(config, dict):
        return [f"Configuration must be a dict, got {type(config).__name__}"]

    for key in REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    for key, allowed in ALLOWED_VALUES.items():
        if key in config and config[key] not in allowed:
            errors.append(f"Invalid value for {key}: {config[key]!r}. Allowed: {sorted(allowed)}")

    numeric_checks = {
        "base_channels": (0, 256),
        "epochs": (1, 200),
        "batch_size": (1, 64),
        "image_size": (64, 1024),
        "lr": (1e-6, 1e-1),
        "cls_weight": (0.1, 10.0),
        "damage_loss_alpha": (0.0, 5.0),
        "early_stopping_patience": (0, 100),
        "early_stopping_min_delta": (0.0, 1.0),
    }
    for key, (lower, upper) in numeric_checks.items():
        if key not in config:
            continue
        value = config[key]
        if not isinstance(value, (int, float)):
            errors.append(f"{key} must be numeric, got {type(value).__name__}")
            continue
        if not (lower <= float(value) <= upper):
            errors.append(f"{key}={value} is outside the supported range [{lower}, {upper}]")

    return errors


def score_config(config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    architectural_score = weighted_average(
        [
            choice_score(config["input_mode"], {"after_only": 0.90, "siamese": 1.00}),
            choice_score(config["encoder_mode"], {"shared": 1.00, "separate": 0.82}),
            choice_score(
                config["fusion_strategy"],
                {"concat": 0.96, "absdiff": 0.78, "concat_absdiff": 1.00},
            ),
            choice_score(config["fusion_stage"], {"bottleneck_only": 1.00, "multiscale": 0.94}),
        ],
        [0.22, 0.18, 0.22, 0.15],
    )

    training_score = weighted_average(
        [
            gaussian_score(float(config["base_channels"]), target=32.0, sigma=14.0),
            log_gaussian_score(float(config["lr"]), target=3e-4, sigma=0.55),
            gaussian_score(float(config["cls_weight"]), target=2.0, sigma=0.7),
            gaussian_score(float(config["epochs"]), target=100.0, sigma=35.0),
            choice_score(config["selection_metric"], {"joint": 1.00, "dice": 0.95, "cls_f1": 0.92, "cls_acc": 0.88}),
        ],
        [0.24, 0.22, 0.20, 0.20, 0.14],
    )

    regularization_score = weighted_average(
        [
            choice_score(config["damage_loss_mode"], {"ce": 0.96, "ordinal": 1.00, "hierarchical": 0.99}),
            gaussian_score(float(config["damage_loss_alpha"]), target=0.5, sigma=0.45),
        ],
        [0.55, 0.45],
    )

    stability_score = weighted_average(
        [
            gaussian_score(float(config["batch_size"]), target=8.0, sigma=4.5),
            patience_score(int(config["early_stopping_patience"])),
            min_delta_score(float(config["early_stopping_min_delta"])),
        ],
        [0.42, 0.32, 0.26],
    )

    # Strong historical family bonus without making the landscape trivial.
    family_bonus = 0.04 if config["input_mode"] == "siamese" and config["encoder_mode"] == "shared" else 0.0
    family_bonus += 0.03 if config["fusion_strategy"] == "concat_absdiff" else 0.0
    family_bonus += 0.02 if config["fusion_stage"] == "bottleneck_only" else 0.0

    combined_score = (
        0.34 * architectural_score
        + 0.26 * training_score
        + 0.14 * regularization_score
        + 0.12 * stability_score
        + 0.14
        + family_bonus
    )

    return combined_score, {
        "architectural_score": architectural_score,
        "training_score": training_score,
        "regularization_score": regularization_score,
        "stability_score": stability_score,
        "family_bonus": family_bonus,
        "combined_score": combined_score,
    }


def choice_score(value: Any, options: Dict[str, float]) -> float:
    return float(options.get(value, 0.0))


def gaussian_score(value: float, target: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-((value - target) ** 2) / (2.0 * sigma**2))


def log_gaussian_score(value: float, target: float, sigma: float) -> float:
    if value <= 0 or target <= 0:
        return 0.0
    return math.exp(-((math.log(value) - math.log(target)) ** 2) / (2.0 * sigma**2))


def weighted_average(values: List[float], weights: List[float]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def patience_score(patience: int) -> float:
    if patience <= 0:
        return 0.86
    if patience <= 4:
        return 0.92
    if patience <= 8:
        return 1.00
    if patience <= 12:
        return 0.98
    return 0.94


def min_delta_score(min_delta: float) -> float:
    if min_delta == 0.0:
        return 1.00
    if min_delta <= 0.01:
        return 0.98
    return 0.94


def error_result(
    messages: List[str],
    start_time: float,
    *,
    config: Dict[str, Any] | None = None,
    status: str = "ERROR",
    extra_artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {
        "status": status,
        "validation_errors": "\n".join(messages),
        "eval_time": f"{time.time() - start_time:.3f}s",
    }
    if config is not None:
        artifacts["candidate_config"] = json.dumps(config, indent=2, sort_keys=True)
    if extra_artifacts:
        artifacts.update(extra_artifacts)

    return {
        "combined_score": 0.0,
        "architectural_score": 0.0,
        "training_score": 0.0,
        "regularization_score": 0.0,
        "stability_score": 0.0,
        "eval_time": time.time() - start_time,
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("program_path")
    args = parser.parse_args()

    print(json.dumps(evaluate(args.program_path), indent=2))
