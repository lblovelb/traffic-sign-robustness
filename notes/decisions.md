# Decisions

Short records of choices and why they were made. Newest at the bottom.

## 2026-11-11 — Dataset: GTSRB
Public, well-known, 43 classes, small images (fast on a laptop CPU). It is
cropped signs only, so this project studies the *classification* step, not
detection. Stated as a limitation.

## 2026-11-11 — Baseline: HOG + linear SVM
The "existing solution" to compare against. Classical traffic-sign pipelines
used hand-crafted gradient features; this shows whether the CNN's advantage
holds when images degrade.

## 2026-11-11 — Synthetic corruptions, severities 1–5
Real bad-weather data with labels is scarce, so corruptions are simulated (in
the spirit of ImageNet-C). Each is seeded so every model sees the same images.

## 2026-11-11 — Seen vs unseen corruptions
Augmentation uses noise, darkness and fog only. Blur and rain are held out to
test whether robustness *generalises* to conditions never trained on — the
question that matters when a vehicle meets something new.

## 2026-11-11 — Three seeds, report mean ± std
One run can be lucky. Same practice as Love's bachelor thesis.

## 2026-11-11 — Shuffle before validation split
GTSRB is stored sorted by class and Keras `validation_split` takes the last 10%.
Shuffle the training data once with a fixed seed (0) so every class appears in
both training and validation. See [[experiment-log]] quick run #1.

## 2026-11-11 — Quick run trains for 10 epochs
2 epochs was too few to learn anything or to see learning curves. 10 epochs
takes ~40 s per CNN in the quick run and matches the full run.

## 2026-11-11 — BatchNorm momentum 0.9
The default 0.99 made the running mean/variance lag behind the weights in short
runs, so the model behaved differently at inference (98% train, ~5% val). 0.9
fixes this and costs nothing in the full run. See quick run #3.

## 2026-11-12 — Learning curves drawn automatically
Per-epoch metrics are already logged to MLflow, so `curves.py` reads them back
and `experiment.py` draws `results/learning_curves.png` at the end of every run.
MLflow only runs locally, so a PNG is the way to show the curves on GitHub.

