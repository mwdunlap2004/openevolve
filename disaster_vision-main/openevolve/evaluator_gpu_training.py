#!/usr/bin/env python3
"""
GPU-backed evaluator for disaster-vision OpenEvolve runs.

Each candidate program returns a training recipe. This evaluator runs a short
real PyTorch training job for that recipe and scores it from validation metrics.
The score equally weights segmentation and classification-oriented metrics:

combined_score = mean(best_val_joint_score, best_val_dice, best_val_iou, best_val_cls_f1)

This is intentionally expensive compared with the surrogate evaluator. Use a
small OpenEvolve iteration count and parallel_evaluations=1.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = ROOT / "jobs"
TRAINING_SCRIPT = JOBS_ROOT / "change_family_experiments.py"
RUNS_ROOT = Path(os.environ.get("OE_TRAINING_RUNS_ROOT", JOBS_ROOT / "runs" / "openevolve_gpu_trials"))

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
    run_dir: Path | None = None
    try:
        config = load_candidate_config(program_path)
        validation_errors = validate_config(config)
        if validation_errors:
            return error_result(validation_errors, start_time, config=config, status="VALIDATION_ERROR")

        train_root = Path(os.environ.get("TRAIN_ROOT", Path.home() / "data" / "xview2_jpeg" / "tier1"))
        val_root = Path(os.environ.get("VAL_ROOT", Path.home() / "data" / "xview2_jpeg" / "hold"))
        data_errors = validate_split_root(train_root, "TRAIN_ROOT") + validate_split_root(val_root, "VAL_ROOT")
        if data_errors:
            return error_result(data_errors, start_time, config=config, status="DATA_ERROR")

        effective_config = make_effective_config(config)
        run_name = make_run_name(effective_config)
        run_dir = RUNS_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        command = build_training_command(effective_config, train_root, val_root, run_name)
        command_path = run_dir / "openevolve_training_command.json"
        command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")

        timeout = int(os.environ.get("OE_TRAINING_TIMEOUT_SECONDS", "7200"))
        completed = subprocess.run(
            command,
            cwd=str(ROOT.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        train_log = run_dir / "openevolve_subprocess.log"
        train_log.write_text(completed.stdout, encoding="utf-8")

        if completed.returncode != 0:
            return error_result(
                [f"Training command failed with exit code {completed.returncode}"],
                start_time,
                config=config,
                status="TRAINING_ERROR",
                extra_artifacts={
                    "run_dir": str(run_dir),
                    "train_log_tail": completed.stdout[-4000:],
                    "command": json.dumps(command, indent=2),
                },
            )

        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return error_result(
                [f"Training finished but metrics.csv was not found: {metrics_path}"],
                start_time,
                config=config,
                status="METRICS_ERROR",
                extra_artifacts={"run_dir": str(run_dir), "train_log_tail": completed.stdout[-4000:]},
            )

        score, best_metrics = score_metrics(metrics_path)
        elapsed = time.time() - start_time

        return {
            "combined_score": float(score),
            "val_joint_score": float(best_metrics["val_joint_score"]),
            "val_dice": float(best_metrics["val_dice"]),
            "val_iou": float(best_metrics["val_iou"]),
            "val_cls_f1": float(best_metrics["val_cls_f1"]),
            "best_epoch": float(best_metrics["epoch"]),
            "eval_time": elapsed,
            "artifacts": {
                "status": "TRAINED",
                "candidate_config": json.dumps(config, indent=2, sort_keys=True),
                "effective_config": json.dumps(effective_config, indent=2, sort_keys=True),
                "score_formula": "equal mean of best val_joint_score, val_dice, val_iou, val_cls_f1",
                "best_metrics": json.dumps(best_metrics, indent=2, sort_keys=True),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "train_log": str(run_dir / "train.log"),
                "subprocess_log": str(run_dir / "openevolve_subprocess.log"),
                "eval_time": f"{elapsed:.3f}s",
            },
        }
    except subprocess.TimeoutExpired as exc:
        return error_result(
            [f"Training timed out after {exc.timeout}s"],
            start_time,
            status="TIMEOUT",
            extra_artifacts={"run_dir": str(run_dir) if run_dir else "", "output_tail": (exc.stdout or "")[-4000:]},
        )
    except Exception as exc:
        return error_result(
            [f"{type(exc).__name__}: {exc}"],
            start_time,
            status="EXCEPTION",
            extra_artifacts={"traceback": traceback.format_exc(), "run_dir": str(run_dir) if run_dir else ""},
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

    raise AttributeError("Program must define run_experiment(), build_experiment_config(), or configure_experiment()")


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
        "base_channels": (8, 96),
        "epochs": (1, 50),
        "batch_size": (1, 16),
        "image_size": (128, 384),
        "lr": (1e-5, 5e-3),
        "cls_weight": (0.25, 5.0),
        "damage_loss_alpha": (0.0, 2.0),
        "early_stopping_patience": (0, 25),
        "early_stopping_min_delta": (0.0, 0.1),
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

    if config.get("input_mode") == "after_only":
        # The trainer forces these anyway; make the constraint explicit to reduce wasted trials.
        if config.get("encoder_mode") != "shared":
            errors.append("after_only candidates must use encoder_mode='shared'")
        if config.get("fusion_stage") != "bottleneck_only":
            errors.append("after_only candidates must use fusion_stage='bottleneck_only'")

    return errors


def validate_split_root(root: Path, env_name: str) -> List[str]:
    errors: List[str] = []
    if not root.exists():
        return [f"{env_name} does not exist: {root}"]
    if not (root / "images_jpeg").exists():
        errors.append(f"{env_name} missing images_jpeg folder: {root / 'images_jpeg'}")
    if not (root / "labels").exists():
        errors.append(f"{env_name} missing labels folder: {root / 'labels'}")
    return errors


def make_effective_config(config: Dict[str, Any]) -> Dict[str, Any]:
    effective = dict(config)
    effective["epochs"] = int(os.environ.get("OE_TRAINING_EPOCHS", "10"))
    effective["debug_samples"] = int(os.environ.get("OE_DEBUG_SAMPLES", "0"))
    effective["num_workers"] = int(os.environ.get("OE_TRAINING_NUM_WORKERS", "4"))
    effective["selection_metric"] = "joint"
    effective["early_stopping_patience"] = min(int(effective.get("early_stopping_patience", 8)), effective["epochs"])
    return effective


def make_run_name(config: Dict[str, Any]) -> str:
    raw_name = str(config.get("experiment_name", "candidate"))
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw_name)[:80]
    return f"oe_gpu_{safe_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def build_training_command(config: Dict[str, Any], train_root: Path, val_root: Path, run_name: str) -> List[str]:
    return [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--train_root",
        str(train_root),
        "--val_root",
        str(val_root),
        "--save_dir",
        str(RUNS_ROOT),
        "--run_name",
        run_name,
        "--exp_name",
        str(config["experiment_name"]),
        "--epochs",
        str(int(config["epochs"])),
        "--batch_size",
        str(int(config["batch_size"])),
        "--image_size",
        str(int(config["image_size"])),
        "--lr",
        str(float(config["lr"])),
        "--num_workers",
        str(int(config["num_workers"])),
        "--base_channels",
        str(int(config["base_channels"])),
        "--cls_weight",
        str(float(config["cls_weight"])),
        "--damage_loss_mode",
        str(config["damage_loss_mode"]),
        "--damage_loss_alpha",
        str(float(config["damage_loss_alpha"])),
        "--debug_samples",
        str(int(config["debug_samples"])),
        "--selection_metric",
        str(config["selection_metric"]),
        "--early_stopping_patience",
        str(int(config["early_stopping_patience"])),
        "--early_stopping_min_delta",
        str(float(config["early_stopping_min_delta"])),
        "--input_mode",
        str(config["input_mode"]),
        "--encoder_mode",
        str(config["encoder_mode"]),
        "--fusion_strategy",
        str(config["fusion_strategy"]),
        "--fusion_stage",
        str(config["fusion_stage"]),
    ]


def score_metrics(metrics_path: Path) -> tuple[float, Dict[str, float]]:
    rows: List[Dict[str, str]] = []
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    if not rows:
        raise ValueError(f"No rows found in metrics file: {metrics_path}")

    required = ["epoch", "val_joint_score", "val_dice", "val_iou", "val_cls_f1"]
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ValueError(f"Metrics file missing columns {missing}: {metrics_path}")

    best_score = -1.0
    best_metrics: Dict[str, float] = {}
    for row in rows:
        metrics = {key: float(row[key]) for key in required}
        score = 0.25 * (
            metrics["val_joint_score"]
            + metrics["val_dice"]
            + metrics["val_iou"]
            + metrics["val_cls_f1"]
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics | {"combined_score": score}

    return best_score, best_metrics


def error_result(
    errors: List[str],
    start_time: float,
    *,
    config: Dict[str, Any] | None = None,
    status: str = "ERROR",
    extra_artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    elapsed = time.time() - start_time
    artifacts = {
        "status": status,
        "errors": "\n".join(errors),
        "eval_time": f"{elapsed:.3f}s",
    }
    if config is not None:
        artifacts["candidate_config"] = json.dumps(config, indent=2, sort_keys=True)
    if extra_artifacts:
        artifacts.update({key: str(value) for key, value in extra_artifacts.items()})
    return {
        "combined_score": 0.0,
        "val_joint_score": 0.0,
        "val_dice": 0.0,
        "val_iou": 0.0,
        "val_cls_f1": 0.0,
        "eval_time": elapsed,
        "artifacts": artifacts,
    }
