# Disaster Vision OpenEvolve Task

This directory contains a self-contained OpenEvolve problem for the
`disaster_vision-main` project.

## What it evolves

The program in `initial_program.py` returns a training recipe for the
change-family experiments in `jobs/change_family_experiments.py`.

The evaluator in `evaluator.py` is a surrogate scorer because the private xView2
dataset is not bundled with this repository. It ranks recipes using the
historical patterns already present in `jobs/leaderboard.csv`.

## Run

From the repository root:

```bash
python openevolve-run.py \
  disaster_vision-main/openevolve/initial_program.py \
  disaster_vision-main/openevolve/evaluator.py \
  --config disaster_vision-main/openevolve/config.yaml \
  --iterations 60
```

## Notes

- The starting recipe is the `after_only` baseline.
- The search space is centered on the strongest observed change-detection
  families in `jobs/leaderboard.csv`.
- If you later want a real training loop instead of the surrogate evaluator,
  this folder is the place to swap in a dataset-backed evaluator.
