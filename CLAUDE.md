# CLAUDE.md — traffic-sign-robustness

Instructions for Claude (and any agent) working in this repo. Read this first.

## What this project is
A portfolio prototype: how robust is traffic-sign recognition when the camera
image degrades (fog, rain, darkness, blur, sensor noise)? We compare a CNN
(TensorFlow/Keras) with a classical HOG + linear SVM baseline, and test whether
training on corrupted images (augmentation) helps — including on corruption
types the model never saw during training.

The point is the **evaluation**, not a high clean accuracy. Every claim in the
README must be backed by a number in `results/`.

## Working with Love
- Love is a junior developer learning this workflow. Explain *why* when you
  make a non-obvious choice, briefly. Talk to him in Swedish; code, comments,
  commits and docs are in English.
- Love must be able to explain every part of this code in a job interview.
  Prefer simple, readable code over clever code.
- Ask before: adding a dependency, changing the experimental design in
  `configs/experiment.yaml`, deleting anything, or pushing to GitHub.

## Layout
- `src/tsr/corruptions.py` — NumPy-only corruptions, severities 1–5, seeded.
- `src/tsr/data.py` — GTSRB download + cache (`data/`, git-ignored).
- `src/tsr/models.py` — `cnn` and `hog_svm`. TensorFlow is imported lazily.
- `src/tsr/experiment.py` — runs all conditions, logs to MLflow + `results/results.csv`.
- `src/tsr/report.py` — summary table + robustness plot from the CSV.
- `src/tsr/curves.py` — learning curves from MLflow (also run by experiment.py).
- `src/tsr/preview.py` — example images of the corruptions.
- `notebooks/analysis.ipynb` — exploratory analysis of `results/results.csv`.
- `configs/experiment.yaml` — **all** experiment settings live here.
- `notes/` — the project vault (Obsidian-compatible). See `notes/README.md`.
- `.claude/agents/` — specialised agents. `.claude/skills/` — repeatable procedures.

## Commands
```bash
pip install -r requirements.txt     # full env (TensorFlow, MLflow)
pip install -e .                    # makes `import tsr` work
pytest                              # tests (no TensorFlow needed)
ruff check . && ruff format --check .
python -m tsr.experiment --quick    # smoke test, minutes
python -m tsr.experiment            # full run, ~1 hour on a laptop CPU
python -m tsr.curves                # redraw learning curves without re-training
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## Rules for experiments (these keep the results trustworthy)
1. Never tune anything on the test set. Validation comes from a split of the
   training data (`validation_split` in Keras).
2. Every number reported is mean ± std over all seeds in the config.
3. Corrupted test sets depend only on `evaluation.corruption_seed`, so all
   models see identical test images. Don't break this.
4. Augmentation uses only `augmentation.corruptions`; the rest stay *unseen*.
   Don't add an unseen corruption to augmentation without saying so in notes.
5. Run `--quick` before a full run after any code change.
6. After a run, log it in `notes/experiment-log.md` (date, config change, headline numbers).

## Code style
- Python 3.11, type hints, `ruff` for lint + format (config in `pyproject.toml`).
- New corruption → add it to `CORRUPTIONS` and add a test in `tests/test_corruptions.py`.
- Keep functions small and pure where possible; randomness only via an explicit seed.
