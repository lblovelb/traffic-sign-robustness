# Traffic-sign recognition under bad weather — a robustness evaluation

**Question:** a traffic-sign classifier scores ~99% on clean images. What
happens when the camera sees fog, rain, darkness, blur or sensor noise — and
does training on degraded images fix it, even for conditions it never saw?

This is a small, reproducible prototype in the style of applied automotive AI
work: build a model, evaluate it against an existing solution with clear
metrics, stress-test robustness on simulated data, and judge feasibility.

## Experimental design

| Condition | Train on | Test on | Answers |
|---|---|---|---|
| **Baseline** | clean | clean | How good are the models normally? |
| **Shift** | clean | corrupted (5 types × 5 severities) | Where and how fast do they break? |
| **Mitigation** | partly corrupted (noise, darkness, fog) | corrupted | Does augmentation help on *seen* corruptions — and on *unseen* ones (blur, rain)? |

- **Models:** a small CNN (TensorFlow/Keras) vs. a classical **HOG + linear
  SVM** baseline.
- **Data:** [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html), 43 classes,
  images resized to 32×32.
- **Metrics:** accuracy, macro-F1, relative drop vs. clean, latency per image,
  model size.
- **Reliability:** 3 seeds, results as mean ± std. All models are tested on
  identical corrupted images (fixed corruption seed).
- **Tracking:** every run is logged in MLflow; every evaluation is a row in
  `results/results.csv`.

The design mirrors my bachelor thesis, *Empirical evaluation of
English-trained machine learning models for phishing email detection in a
Scandinavian context* (Linnaeus University, 2026), which measured how
models degrade under a distribution shift (language) and whether a
mitigation (translation) recovers performance.

## Results

*Run the experiment to fill this in — see `results/summary.md`.*

<!-- After the full run: headline table, robustness.png, and 3–5 bullet
conclusions written from notes/analysis.md -->

## Run it

```bash
conda create -n tsr python=3.11 -y && conda activate tsr
pip install -r requirements.txt
pip install -e .

pytest                              # tests (no TensorFlow needed)
python -m tsr.experiment --quick    # smoke test, a few minutes
python -m tsr.experiment            # full experiment
mlflow ui                           # browse runs at http://127.0.0.1:5000
```

The dataset (~300 MB) downloads automatically on the first run into `data/`.

## Project layout

```
src/tsr/
  corruptions.py   simulated fog, rain, darkness, blur, noise (NumPy, seeded)
  data.py          GTSRB download + cache
  models.py        CNN and HOG+SVM
  metrics.py       accuracy, macro-F1, latency
  experiment.py    runs all conditions, logs to MLflow + CSV
  report.py        summary table + robustness plot
configs/experiment.yaml   every experimental setting in one place
tests/                    unit tests, run in CI on every push
notes/                    project vault: decisions, experiment log, analysis
```

## How this project was built: AI-assisted workflow

The repo includes the configuration I use to work with Claude Code as a team
of specialised agents. It is versioned like code:

- `CLAUDE.md`: project rules that every agent follows, for example never
  tuning on the test set and reporting every number as mean ± std over seeds.
- `.claude/agents/`
  - `code-reviewer`: a read-only reviewer with a dedicated
    *experimental-validity* pass (data leakage, seeds, unseen corruptions
    leaking into training). Findings are confidence-scored and anything below
    80/100 is dropped.
  - `experiment-runner`: runs the smoke test before the full run and logs
    each run.
  - `results-analyst`: writes conclusions that are backed by the numbers and
    states the limitations.
- `.claude/skills/run-robustness-eval`: chains the agents in order: lint/test,
  then review, then run, then analyse.
- `notes/`: an Obsidian-compatible vault that serves as long-term memory
  across sessions.

## Limitations

- The corruptions are synthetic. They approximate weather and are not real
  recordings of it.
- GTSRB contains cropped signs, so this covers classification only and not
  detection in full scenes.
- Results come from a single dataset and a single image resolution.
