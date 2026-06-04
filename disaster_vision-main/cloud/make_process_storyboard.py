#!/usr/bin/env python3
"""Collect process artifacts for a demo video or slide walkthrough.

The script is intentionally read-only against run outputs. It copies existing
plots, creates simple progress frames, and writes a storyboard/manifest that can
be used to narrate how OpenEvolve proposes recipes and how real model training
validates them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disaster-vision process artifacts")
    parser.add_argument("--training-run", type=Path, default=None)
    parser.add_argument("--openevolve-output", type=Path, default=ROOT / "openevolve" / "openevolve_output")
    parser.add_argument("--segformer-outputs", type=Path, default=ROOT / "SegFormer" / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "process_artifacts")
    parser.add_argument("--make-video", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    evidence_dir = output_dir / "evidence"
    frames_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    notes: list[str] = []

    training_run = args.training_run or find_latest_training_run(ROOT / "jobs" / "runs")
    checkpoint = find_latest_checkpoint(args.openevolve_output)

    add_title_frame(
        frames_dir / "000_overview.png",
        "Disaster Vision Optimization",
        [
            "Goal: evolve a stronger change-detection training recipe.",
            "OpenEvolve searches recipe choices with a surrogate evaluator.",
            "Google Cloud GPU training validates selected recipes on xView2 data.",
        ],
    )
    manifest.append(frame_row("000_overview.png", "Overview", "Opening slide for the video."))

    if checkpoint:
        evo_summary = summarize_checkpoint(checkpoint, output_dir)
        plot_evolution_scores(evo_summary["programs"], frames_dir / "010_openevolve_scores.png")
        manifest.append(
            frame_row(
                "010_openevolve_scores.png",
                "OpenEvolve progress",
                "Candidate recipe scores from the latest checkpoint.",
            )
        )
        notes.extend(format_evolution_notes(checkpoint, evo_summary))
    else:
        add_title_frame(
            frames_dir / "010_openevolve_scores.png",
            "OpenEvolve Progress",
            ["No checkpoint found yet.", "Run the OpenEvolve cloud loop, then rerun this capture script."],
        )
        manifest.append(frame_row("010_openevolve_scores.png", "OpenEvolve progress", "Placeholder."))
        notes.append("- OpenEvolve checkpoint: not found.")

    if training_run:
        copied = copy_training_artifacts(training_run, evidence_dir, frames_dir)
        plot_training_metrics(training_run / "metrics.csv", frames_dir / "020_training_metrics.png")
        manifest.append(
            frame_row(
                "020_training_metrics.png",
                "Training validation",
                "Validation metrics from the selected cloud training run.",
            )
        )
        notes.extend(format_training_notes(training_run, copied))
    else:
        add_title_frame(
            frames_dir / "020_training_metrics.png",
            "Training Validation",
            ["No training run found yet.", "Run cloud training, then rerun this capture script."],
        )
        manifest.append(frame_row("020_training_metrics.png", "Training validation", "Placeholder."))
        notes.append("- Training run: not found.")

    segformer_copied = copy_segformer_examples(args.segformer_outputs, evidence_dir, frames_dir)
    if segformer_copied:
        manifest.append(
            frame_row(
                "030_model_examples.png",
                "Model examples",
                "Representative prediction or diagnostic image copied from model outputs.",
            )
        )
        notes.append(f"- Model example evidence copied: {len(segformer_copied)} file(s).")
    else:
        add_title_frame(
            frames_dir / "030_model_examples.png",
            "Model Examples",
            ["No prediction examples found yet.", "Add prediction or failure/success PNGs to run outputs."],
        )
        manifest.append(frame_row("030_model_examples.png", "Model examples", "Placeholder."))
        notes.append("- Model example evidence: not found.")

    write_manifest(output_dir / "manifest.csv", manifest)
    write_storyboard(output_dir / "storyboard.md", notes, manifest)
    write_summary(output_dir / "summary.json", training_run, checkpoint, manifest)

    if args.make_video:
        make_video(frames_dir, output_dir / "disaster_vision_process.mp4")

    print(f"Process artifacts written to: {output_dir}")
    print(f"Storyboard: {output_dir / 'storyboard.md'}")
    print(f"Frames: {frames_dir}")


def find_latest_training_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "metrics.csv").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "metrics.csv").stat().st_mtime)


def find_latest_checkpoint(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    if output_root.name.startswith("checkpoint_"):
        return output_root
    checkpoints = [p for p in output_root.rglob("checkpoint_*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def summarize_checkpoint(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    programs_dir = checkpoint / "programs"
    programs: list[dict[str, Any]] = []
    if programs_dir.exists():
        for path in sorted(programs_dir.glob("*.json")):
            with path.open() as handle:
                program = json.load(handle)
            metrics = program.get("metrics") or {}
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)):
                programs.append(
                    {
                        "id": program.get("id", path.stem),
                        "score": float(score),
                        "metrics": metrics,
                        "artifacts_json": program.get("artifacts_json"),
                        "code": program.get("code", ""),
                    }
                )

    programs.sort(key=lambda item: item["score"], reverse=True)
    examples_path = output_dir / "openevolve_candidate_examples.md"
    with examples_path.open("w") as handle:
        handle.write("# OpenEvolve candidate examples\n\n")
        if not programs:
            handle.write("No scored programs found in checkpoint.\n")
        for label, item in (("Best", programs[0] if programs else None), ("Lowest scored", programs[-1] if programs else None)):
            if item is None:
                continue
            handle.write(f"## {label}: {item['id']}\n\n")
            handle.write(f"- combined_score: {item['score']:.4f}\n")
            handle.write("- metrics:\n\n")
            handle.write("```json\n")
            handle.write(json.dumps(item["metrics"], indent=2, sort_keys=True))
            handle.write("\n```\n\n")
            if item.get("artifacts_json"):
                handle.write("- evaluator artifacts:\n\n")
                handle.write("```json\n")
                handle.write(str(item["artifacts_json"])[:4000])
                handle.write("\n```\n\n")
            if item.get("code"):
                handle.write("- candidate code excerpt:\n\n")
                handle.write("```python\n")
                handle.write(item["code"][:4000])
                handle.write("\n```\n\n")

    return {"checkpoint": checkpoint, "programs": programs, "examples_path": examples_path}


def plot_evolution_scores(programs: list[dict[str, Any]], output_path: Path) -> None:
    if not programs:
        add_title_frame(output_path, "OpenEvolve Scores", ["No scored candidates found."])
        return
    plt = get_plt()
    scores = [item["score"] for item in sorted(programs, key=lambda item: item["score"])]
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(scores) + 1), scores, marker="o", linewidth=2)
    plt.title("OpenEvolve Candidate Score Distribution")
    plt.xlabel("Candidate rank")
    plt.ylabel("combined_score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_training_metrics(metrics_path: Path, output_path: Path) -> None:
    if not metrics_path.exists():
        add_title_frame(output_path, "Training Metrics", ["metrics.csv was not found."])
        return
    rows = read_csv_rows(metrics_path)
    if not rows:
        add_title_frame(output_path, "Training Metrics", ["metrics.csv had no rows."])
        return
    plt = get_plt()
    epochs = [int(float(row["epoch"])) for row in rows]
    plt.figure(figsize=(12, 7))
    for column in ("val_dice", "val_iou", "val_cls_f1", "val_joint_score"):
        if column in rows[0]:
            plt.plot(epochs, [float(row[column]) for row in rows], marker="o", label=column)
    plt.title("Cloud Training Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def copy_training_artifacts(training_run: Path, evidence_dir: Path, frames_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for source in [
        training_run / "config.json",
        training_run / "metrics.csv",
        training_run / "plots" / "loss_curves.png",
        training_run / "plots" / "segmentation_metrics.png",
        training_run / "plots" / "classification_metrics.png",
        training_run / "train.log",
    ]:
        if source.exists():
            dest = evidence_dir / f"training_{source.name}"
            shutil.copy2(source, dest)
            copied.append(dest)

    representative = training_run / "plots" / "classification_metrics.png"
    if representative.exists():
        shutil.copy2(representative, frames_dir / "021_training_classification_metrics.png")
    return copied


def copy_segformer_examples(segformer_outputs: Path, evidence_dir: Path, frames_dir: Path) -> list[Path]:
    copied: list[Path] = []
    candidates = [
        "prediction_example.png",
        "confusion_matrix_building_pixels.png",
        "training_curves.png",
        "per_class_iou_over_epochs.png",
    ]
    first_image: Path | None = None
    for name in candidates:
        source = segformer_outputs / name
        if source.exists():
            dest = evidence_dir / f"segformer_{name}"
            shutil.copy2(source, dest)
            copied.append(dest)
            first_image = first_image or source
    if first_image:
        shutil.copy2(first_image, frames_dir / "030_model_examples.png")
    return copied


def format_evolution_notes(checkpoint: Path, summary: dict[str, Any]) -> list[str]:
    programs = summary["programs"]
    notes = [f"- OpenEvolve checkpoint: `{checkpoint}`."]
    if programs:
        notes.append(f"- Scored candidates captured: {len(programs)}.")
        notes.append(f"- Best candidate score: {programs[0]['score']:.4f}.")
        notes.append(f"- Lowest captured candidate score: {programs[-1]['score']:.4f}.")
        notes.append(f"- Candidate examples: `{summary['examples_path']}`.")
    return notes


def format_training_notes(training_run: Path, copied: list[Path]) -> list[str]:
    notes = [f"- Training run: `{training_run}`."]
    metrics_path = training_run / "metrics.csv"
    rows = read_csv_rows(metrics_path) if metrics_path.exists() else []
    if rows and "val_joint_score" in rows[0]:
        best = max(float(row["val_joint_score"]) for row in rows)
        notes.append(f"- Best validation joint score: {best:.4f}.")
    notes.append(f"- Training evidence copied: {len(copied)} file(s).")
    return notes


def add_title_frame(output_path: Path, title: str, lines: list[str]) -> None:
    plt = get_plt()
    plt.figure(figsize=(12, 7))
    plt.axis("off")
    plt.text(0.05, 0.82, title, fontsize=30, weight="bold", va="top")
    y = 0.62
    for line in lines:
        plt.text(0.08, y, line, fontsize=17, va="top")
        y -= 0.12
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frame_row(filename: str, title: str, narration: str) -> dict[str, str]:
    return {"frame": filename, "title": title, "narration": narration}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "title", "narration"])
        writer.writeheader()
        writer.writerows(rows)


def write_storyboard(path: Path, notes: list[str], manifest: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        handle.write("# Disaster Vision process storyboard\n\n")
        handle.write("## Evidence summary\n\n")
        handle.write("\n".join(notes) if notes else "- No evidence found yet.")
        handle.write("\n\n## Suggested video sequence\n\n")
        for row in manifest:
            handle.write(f"- `{row['frame']}`: {row['title']} - {row['narration']}\n")


def write_summary(path: Path, training_run: Path | None, checkpoint: Path | None, manifest: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        json.dump(
            {
                "training_run": str(training_run) if training_run else None,
                "openevolve_checkpoint": str(checkpoint) if checkpoint else None,
                "frames": manifest,
            },
            handle,
            indent=2,
        )


def make_video(frames_dir: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found; skipping video creation.")
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            "0.5",
            "-pattern_type",
            "glob",
            "-i",
            str(frames_dir / "*.png"),
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
